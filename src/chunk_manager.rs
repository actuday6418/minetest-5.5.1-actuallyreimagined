use crate::frustum::{Aabb, Frustum};
use crate::world::{CHUNK_SIZE, ChunkBlocks, ChunkCoords, World};
use crate::worldgen::{FaceData, generate_chunk_mesh};
use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::{Vec2, f32::Vec3};
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

type MeshGenResult = (ChunkCoords, CpuMeshData);
type BlockGenResult = (ChunkCoords, Arc<ChunkBlocks>);

#[derive(Debug)]
enum ChunkState {
    GeneratingBlocks,
    AwaitingNeighbors,
    Meshing,
    MeshGenerated(CpuMeshData),
    GpuReady(GpuChunkData),
}

pub struct ChunkManager {
    view_radius: i32,
    max_uploads_per_frame: usize,
    chunks: HashMap<ChunkCoords, ChunkState>,
    last_camera_chunk_coords: ChunkCoords,
    world: Arc<RwLock<World>>,
    thread_pool: Arc<ThreadPool>,
    block_result_receiver: Receiver<BlockGenResult>,
    mesh_result_receiver: Receiver<MeshGenResult>,
    block_result_sender: Sender<BlockGenResult>,
    mesh_result_sender: Sender<MeshGenResult>,
}

const DIRECT_NEIGHBOR_OFFSETS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

impl ChunkManager {
    pub fn new(view_radius: i32, max_uploads_per_frame: usize, world: Arc<RwLock<World>>) -> Self {
        let thread_pool = Arc::new(ThreadPoolBuilder::new().num_threads(12).build().unwrap());
        let (mesh_result_sender, mesh_result_receiver) = unbounded::<MeshGenResult>();
        let (block_result_sender, block_result_receiver) =
            unbounded::<(ChunkCoords, Arc<ChunkBlocks>)>();
        Self {
            view_radius,
            max_uploads_per_frame,
            chunks: HashMap::new(),
            last_camera_chunk_coords: (i32::MAX, i32::MAX),
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
        camera_pos_xz: Vec2,
    ) -> impl Iterator<Item = (ChunkCoords, &'a GpuChunkData, f32)> {
        self.chunks
            .iter()
            .filter_map(|(coords, state)| match state {
                ChunkState::GpuReady(gpu_data) if gpu_data.face_buffer.is_some() => {
                    Some((*coords, gpu_data))
                }
                _ => None,
            })
            .filter(|(coords, _gpu_data)| {
                let aabb = Self::chunk_aabb_from_coords(*coords);
                frustum.intersects_aabb(&aabb)
            })
            .map(move |(coords, gpu_data)| {
                let chunk_center_x =
                    coords.0 as f32 * CHUNK_SIZE as f32 + (CHUNK_SIZE as f32 / 2.0);
                let chunk_center_z =
                    coords.1 as f32 * CHUNK_SIZE as f32 + (CHUNK_SIZE as f32 / 2.0);
                let dist_sq =
                    camera_pos_xz.distance_squared(Vec2::new(chunk_center_x, chunk_center_z));
                (coords, gpu_data, dist_sq)
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

                self.check_neighbors_and_schedule_mesh(coord);

                for offset in DIRECT_NEIGHBOR_OFFSETS {
                    let neighbor_coord = (coord.0 + offset.0, coord.1 + offset.1);

                    if self.chunks.contains_key(&neighbor_coord) {
                        self.check_neighbors_and_schedule_mesh(neighbor_coord);
                    }
                }
            } else {
                log::debug!(
                    "Received blocks for untracked or unexpected state chunk: {:?}",
                    coord
                );

                self.world.write().unwrap().remove_chunk_blocks(&coord);
            }
        }
    }

    fn process_mesh_generation_results(&mut self) {
        while let Ok((coord, cpu_mesh_data)) = self.mesh_result_receiver.try_recv() {
            if let Some(state @ ChunkState::Meshing) = self.chunks.get_mut(&coord) {
                *state = ChunkState::MeshGenerated(cpu_mesh_data);
            } else {
                log::debug!(
                    "Received mesh for untracked or unexpected state chunk: {:?}",
                    coord
                );
            }
        }
    }

    fn check_neighbors_and_schedule_mesh(&mut self, coord: ChunkCoords) {
        let can_mesh = {
            if !matches!(self.chunks.get(&coord), Some(ChunkState::AwaitingNeighbors)) {
                false
            } else {
                let world_reader = self.world.read().unwrap();
                DIRECT_NEIGHBOR_OFFSETS.iter().all(|offset| {
                    let neighbor_coord = (coord.0 + offset.0, coord.1 + offset.1);
                    world_reader.chunk_exists(neighbor_coord)
                })
            }
        };

        if can_mesh {
            if let Some(state @ ChunkState::AwaitingNeighbors) = self.chunks.get_mut(&coord) {
                *state = ChunkState::Meshing;

                let world_arc = self.world.clone();
                let mesh_sender = self.mesh_result_sender.clone();

                let neighborhood = {
                    let world_reader = world_arc.read().unwrap();
                    world_reader
                        .get_chunk_neighborhood(coord)
                        .expect("Neighbors checked but neighborhood fetch failed")
                };

                self.thread_pool.spawn(move || {
                    let faces = generate_chunk_mesh(&neighborhood);

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

        let coords_to_upload: Vec<ChunkCoords> = self
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

    fn update_chunk_requests(&mut self, camera_chunk_coords: ChunkCoords) {
        let mut required_coords = HashSet::new();
        for x in
            (camera_chunk_coords.0 - self.view_radius)..=(camera_chunk_coords.0 + self.view_radius)
        {
            for z in (camera_chunk_coords.1 - self.view_radius)
                ..=(camera_chunk_coords.1 + self.view_radius)
            {
                required_coords.insert((x, z));
            }
        }

        let current_coords: HashSet<ChunkCoords> = self.chunks.keys().cloned().collect();

        let coords_to_unload: Vec<_> = current_coords
            .difference(&required_coords)
            .cloned()
            .collect();
        {
            let mut world_writer = self.world.write().unwrap();
            for coord in coords_to_unload {
                log::debug!("Unloading chunk: {:?}", coord);
                self.chunks.remove(&coord);
                world_writer.remove_chunk_blocks(&coord);
            }
        }

        let coords_to_load: Vec<_> = required_coords
            .difference(&current_coords)
            .cloned()
            .collect();
        for coord in coords_to_load {
            log::debug!("Requesting chunk: {:?}", coord);
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
    fn get_chunk_coords_at(position: Vec3) -> ChunkCoords {
        (
            (position.x / CHUNK_SIZE as f32).floor() as i32,
            (position.z / CHUNK_SIZE as f32).floor() as i32,
        )
    }

    #[inline]
    fn chunk_aabb_from_coords(coords: ChunkCoords) -> Aabb {
        let chunk_world_x = coords.0 as f32 * CHUNK_SIZE as f32;
        let chunk_world_z = coords.1 as f32 * CHUNK_SIZE as f32;
        Aabb {
            min: Vec3::new(chunk_world_x, 0.0, chunk_world_z),
            max: Vec3::new(
                chunk_world_x + CHUNK_SIZE as f32,
                CHUNK_SIZE as f32,
                chunk_world_z + CHUNK_SIZE as f32,
            ),
        }
    }
}
