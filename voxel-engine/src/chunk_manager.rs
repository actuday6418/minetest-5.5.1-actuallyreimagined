use crate::frustum::{Aabb, Frustum};
use crate::meshing::CHUNK_SIZE;
use crate::meshing::{build_chunk_mesh, data::FaceData};
use crate::world::{ChunkBlocks, World};
use crate::{CHUNK_RADIUS, CHUNK_RADIUS_VERTICAL};
use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::IVec3;
use glam::f32::Vec3;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct GpuChunkData {
    pub face_buffer: Option<wgpu::Buffer>,
    pub face_count: u32,
}

#[derive(Debug)]
pub struct CpuMeshData {
    pub faces: Vec<FaceData>,
}

type MeshGenResult = (IVec3, CpuMeshData);
type BlockGenResult = (IVec3, Arc<ChunkBlocks>);

#[derive(Debug)]
enum ChunkState {
    GeneratingBlocks,
    AwaitingNeighbors,
    Meshing,
    MeshGenerated(CpuMeshData),
    GpuReady(GpuChunkData),
}

pub struct ChunkManager {
    horiz_radius: i32,
    vert_radius: i32,
    max_uploads_per_frame: usize,
    chunks: HashMap<IVec3, ChunkState>,
    last_camera_chunk_coords: IVec3,
    pub world: Arc<RwLock<World>>,
    thread_pool: Arc<ThreadPool>,
    block_result_receiver: Receiver<BlockGenResult>,
    mesh_result_receiver: Receiver<MeshGenResult>,
    block_result_sender: Sender<BlockGenResult>,
    mesh_result_sender: Sender<MeshGenResult>,
}

const SELF_N_NEIGHBOR_OFFSETS: [IVec3; 7] = [
    IVec3::new(0, 0, 0),  // Self
    IVec3::new(0, 0, 1),  // North
    IVec3::new(0, 0, -1), // South
    IVec3::new(1, 0, 0),  // East
    IVec3::new(-1, 0, 0), // West
    IVec3::new(0, 1, 0),  // Up
    IVec3::new(0, -1, 0), // Down
];

impl ChunkManager {
    pub fn new(
        horiz_radius: i32,
        vert_radius: i32,
        max_uploads_per_frame: usize,
        world: Arc<RwLock<World>>,
    ) -> Self {
        let thread_pool = Arc::new(ThreadPoolBuilder::new().num_threads(12).build().unwrap());
        let (mesh_result_sender, mesh_result_receiver) = unbounded::<MeshGenResult>();
        let (block_result_sender, block_result_receiver) = unbounded::<(IVec3, Arc<ChunkBlocks>)>();
        Self {
            horiz_radius,
            vert_radius,
            max_uploads_per_frame,
            chunks: HashMap::new(),
            last_camera_chunk_coords: IVec3::splat(i32::MAX),
            world,
            thread_pool,
            block_result_receiver,
            mesh_result_receiver,
            block_result_sender,
            mesh_result_sender,
        }
    }

    pub fn update(&mut self, camera_position: Vec3, device: Arc<wgpu::Device>) {
        let camera_chunk_coords = Self::get_chunk_coords_at(camera_position);

        self.process_block_generation_results();
        self.process_mesh_generation_results();

        if camera_chunk_coords != self.last_camera_chunk_coords {
            self.update_chunk_requests(camera_chunk_coords);
            self.last_camera_chunk_coords = camera_chunk_coords;
        }

        self.process_uploads(device);
    }

    pub fn get_renderable_chunks<'a>(
        &'a self,
        frustum: &Frustum,
        camera_pos: Vec3,
    ) -> impl Iterator<Item = (IVec3, &'a GpuChunkData, f32)> {
        self.chunks
            .iter()
            .filter_map(move |(coords, state)| match state {
                ChunkState::GpuReady(gpu_data) if gpu_data.face_buffer.is_some() => {
                    let aabb = Self::chunk_aabb_from_coords(*coords);
                    let chunk_center = coords
                        .as_vec3()
                        .map(|e| e * CHUNK_SIZE as f32 + (CHUNK_SIZE as f32 / 2.0));
                    let dist_sq = camera_pos.distance_squared(chunk_center);

                    frustum
                        .intersects_aabb(&aabb)
                        .then_some((*coords, gpu_data, dist_sq))
                }
                _ => None,
            })
    }

    pub fn _get_chunk_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for state in self.chunks.values() {
            let key = match state {
                ChunkState::GeneratingBlocks => "GeneratingBlocks",
                ChunkState::AwaitingNeighbors => "AwaitingNeighbors",
                ChunkState::Meshing => "Meshing",
                ChunkState::MeshGenerated(_) => "MeshGenerated",
                ChunkState::GpuReady(_) => "GpuReady",
            }
            .to_string();
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn process_block_generation_results(&mut self) {
        while let Ok((coord, chunk_blocks)) = self.block_result_receiver.try_recv() {
            if let Some(state @ ChunkState::GeneratingBlocks) = self.chunks.get_mut(&coord) {
                self.world
                    .write()
                    .unwrap()
                    .insert_chunk_blocks(coord, chunk_blocks);
                *state = ChunkState::AwaitingNeighbors;
                for offset in SELF_N_NEIGHBOR_OFFSETS {
                    let neighbor_coord = coord + offset;
                    if self.chunks.contains_key(&neighbor_coord) {
                        self.check_neighbors_and_schedule_mesh(neighbor_coord);
                    }
                }
            } else {
                self.world.write().unwrap().remove_chunk_blocks(&coord);
            }
        }
    }

    fn process_mesh_generation_results(&mut self) {
        while let Ok((coord, cpu_mesh_data)) = self.mesh_result_receiver.try_recv() {
            if let Some(state @ ChunkState::Meshing) = self.chunks.get_mut(&coord) {
                *state = ChunkState::MeshGenerated(cpu_mesh_data);
            }
        }
    }

    fn check_neighbors_and_schedule_mesh(&mut self, coord: IVec3) {
        let maybe_neighbourhood =
            if !matches!(self.chunks.get(&coord), Some(ChunkState::AwaitingNeighbors)) {
                None
            } else {
                let world_reader = self.world.read().unwrap();
                if SELF_N_NEIGHBOR_OFFSETS.iter().all(|offset| {
                    let neighbor_coord = coord + offset;
                    world_reader.chunk_exists(neighbor_coord)
                }) {
                    Some(
                        world_reader
                            .get_chunk_neighborhood(coord)
                            .expect("Neighbors checked but neighborhood fetch failed"),
                    )
                } else {
                    None
                }
            };

        if let Some(neighborhood) = maybe_neighbourhood {
            if let Some(state @ ChunkState::AwaitingNeighbors) = self.chunks.get_mut(&coord) {
                *state = ChunkState::Meshing;
                let mesh_sender = self.mesh_result_sender.clone();
                self.thread_pool.spawn(move || {
                    let faces = build_chunk_mesh(&neighborhood);

                    let cpu_mesh_data = CpuMeshData { faces };
                    if mesh_sender.send((coord, cpu_mesh_data)).is_err() {
                        log::error!("Mesh result channel closed for {:?}", coord);
                    }
                });
            }
        }
    }

    fn process_uploads(&mut self, device: Arc<wgpu::Device>) {
        let mut uploaded_count = 0;

        let coords_to_upload: Vec<IVec3> = self
            .chunks
            .iter()
            .filter_map(|(coord, state)| {
                if matches!(state, ChunkState::MeshGenerated(_)) {
                    Some(*coord)
                } else {
                    None
                }
            })
            .collect();

        for coord in coords_to_upload {
            if uploaded_count >= self.max_uploads_per_frame {
                break;
            }

            if let Some(ChunkState::MeshGenerated(mesh_data)) = self.chunks.remove(&coord) {
                let face_count = mesh_data.faces.len() as u32;
                let gpu_face_buffer = if face_count > 0 {
                    Some(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("Chunk Face Buffer {:?}", coord)),
                            contents: bytemuck::cast_slice(&mesh_data.faces),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                    )
                } else {
                    None
                };

                let gpu_chunk_data = GpuChunkData {
                    face_buffer: gpu_face_buffer,
                    face_count,
                };

                self.chunks
                    .insert(coord, ChunkState::GpuReady(gpu_chunk_data));
                uploaded_count += 1;
            } else {
                log::error!(
                    "Chunk {:?} was expected to be MeshGenerated but wasn't found or had wrong state during upload processing",
                    coord
                );
            }
        }
    }

    fn update_chunk_requests(&mut self, camera_chunk_coords: IVec3) {
        let mut required_coords = HashSet::new();
        for x in (camera_chunk_coords.x - self.horiz_radius)
            ..=(camera_chunk_coords.x + self.horiz_radius)
        {
            for y in (camera_chunk_coords.y - self.vert_radius)
                ..=(camera_chunk_coords.y + self.vert_radius)
            {
                for z in (camera_chunk_coords.z - self.horiz_radius)
                    ..=(camera_chunk_coords.z + self.horiz_radius)
                {
                    let chunk_coord = IVec3::new(x, y, z);
                    let dcoord = chunk_coord - camera_chunk_coords;

                    if dcoord.x <= CHUNK_RADIUS
                        && dcoord.y <= CHUNK_RADIUS_VERTICAL
                        && dcoord.z <= CHUNK_RADIUS
                    {
                        required_coords.insert(chunk_coord);
                    }
                }
            }
        }

        let current_coords: HashSet<IVec3> = self.chunks.keys().cloned().collect();

        let coords_to_unload: Vec<_> = current_coords
            .difference(&required_coords)
            .cloned()
            .collect();
        {
            let mut world_writer = self.world.write().unwrap();
            for coord in coords_to_unload {
                self.chunks.remove(&coord);
                world_writer.remove_chunk_blocks(&coord);
            }
        }

        let coords_to_load: Vec<_> = required_coords
            .difference(&current_coords)
            .cloned()
            .collect();
        for coord in coords_to_load {
            self.chunks.insert(coord, ChunkState::GeneratingBlocks);

            let block_sender = self.block_result_sender.clone();
            self.thread_pool.spawn(move || {
                let chunk_blocks = Arc::new(ChunkBlocks::generate(coord));
                if block_sender.send((coord, chunk_blocks)).is_err() {
                    log::error!("Block result channel closed for {:?}", coord);
                }
            });
        }
    }

    #[inline]
    fn get_chunk_coords_at(position: Vec3) -> IVec3 {
        position.map(|e| (e / CHUNK_SIZE as f32).floor()).as_ivec3()
    }

    #[inline]
    fn chunk_aabb_from_coords(coords: IVec3) -> Aabb {
        let chunk_coords_global = coords.map(|e| e * CHUNK_SIZE as i32).as_vec3();
        Aabb {
            min: chunk_coords_global,
            max: chunk_coords_global + CHUNK_SIZE as f32,
        }
    }
}
