use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use std::collections::HashMap;
use std::sync::Arc;

pub type ChunkCoords = (i32, i32, i32);

pub const CHUNK_SIZE: usize = 64;

#[derive(Clone)]
pub struct ChunkBlocks {
    blocks: [[[Option<BlockType>; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum BlockType {
    _Water,
    Dirt,
    Grass,
    Stone,
}

impl BlockType {
    pub fn is_opaque(&self) -> bool {
        match self {
            BlockType::_Water => false,
            _ => true,
        }
    }
}

enum BiomeType {
    Plains,
    Hills,
    Mountains,
}

impl ChunkBlocks {
    fn new_empty() -> Self {
        Self {
            blocks: [[[None; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
        }
    }

    pub fn generate(coords: ChunkCoords) -> Self {
        let mut chunk = Self::new_empty();

        const SEED: u32 = 1;
        const SEA_LEVEL: f64 = 512.0;
        const BASE_HORIZONTAL_SCALE: f64 = 0.01;

        const REGION_NOISE_SCALE: f64 = 0.02;
        const MIN_VERTICAL_SCALE: f64 = 5.0;
        const MAX_VERTICAL_SCALE: f64 = 45.0;
        const MIN_VERT_EFFECT_SCALE: f64 = 0.05;
        const MAX_VERT_EFFECT_SCALE: f64 = 0.8;

        const DENSITY_H_SCALE: f64 = 0.02;
        const DENSITY_V_SCALE: f64 = 0.02;

        const MIN_DENSITY_WEIGHT: f64 = 3.0;
        const MAX_DENSITY_WEIGHT: f64 = 120.0;
        const DENSITY_THRESHOLD: f64 = 0.0;
        const CAVE_FLOOR_DEPTH: f64 = 5.0;

        const CAVE_H_SCALE: f64 = 0.012;
        const CAVE_V_SCALE: f64 = 0.028;
        const CAVE_THRESHOLD: f64 = 0.2;
        const CAVE_MIN_Y: f64 = 5.0;
        const CAVE_MAX_Y: f64 = SEA_LEVEL - 5.0;
        const CAVE_SURFACE_BUFFER: f64 = 8.0;

        let heightmap = Fbm::<Perlin>::new(SEED)
            .set_octaves(2)
            .set_frequency(1.0)
            .set_lacunarity(2.0)
            .set_persistence(0.5);
        let weirdness_noise = Perlin::new(SEED.wrapping_add(2));
        let verticality = Perlin::new(SEED.wrapping_add(1));
        let region_noise = Perlin::new(SEED.wrapping_add(3));
        let density_noise = Fbm::<Perlin>::new(SEED.wrapping_add(4))
            .set_frequency(1.0)
            .set_octaves(4)
            .set_lacunarity(2.0)
            .set_persistence(0.5);
        let cave_noise = Fbm::<Perlin>::new(SEED.wrapping_add(6))
            .set_frequency(1.0)
            .set_octaves(4)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = coords.0 as f64 * CHUNK_SIZE as f64 + x as f64;
                let world_z = coords.2 as f64 * CHUNK_SIZE as f64 + z as f64;

                let region_nx = world_x * REGION_NOISE_SCALE;
                let region_nz = world_z * REGION_NOISE_SCALE;
                let region_value = region_noise.get([region_nx, region_nz]);
                let weirdness = weirdness_noise.get([region_nx / 4.0, region_nz / 4.0]);
                let region_factor = ((region_value + 1.0) / 2.0).clamp(0.0, 1.0);
                let smoothed_region_factor =
                    region_factor * region_factor * (3.0 - 2.0 * region_factor);
                let _current_biome = if smoothed_region_factor < 0.3 {
                    BiomeType::Plains
                } else if smoothed_region_factor < 0.7 {
                    BiomeType::Hills
                } else {
                    BiomeType::Mountains
                };

                let density_weight = weirdness
                    * (MIN_DENSITY_WEIGHT
                        + smoothed_region_factor * (MAX_DENSITY_WEIGHT - MIN_DENSITY_WEIGHT));
                let current_vertical_scale = MIN_VERTICAL_SCALE
                    + smoothed_region_factor * (MAX_VERTICAL_SCALE - MIN_VERTICAL_SCALE);
                let current_vert_effect_scale = MIN_VERT_EFFECT_SCALE
                    + smoothed_region_factor * (MAX_VERT_EFFECT_SCALE - MIN_VERT_EFFECT_SCALE);

                let nx_base = world_x * BASE_HORIZONTAL_SCALE;
                let nz_base = world_z * BASE_HORIZONTAL_SCALE;
                let h_base = heightmap.get([nx_base, nz_base]);
                let v = verticality.get([nx_base + 0.1, nz_base + 0.1]);
                let verticality_multiplier = 1.0 + v * current_vert_effect_scale;
                let h_modified = h_base * verticality_multiplier;
                let approx_world_height = SEA_LEVEL + h_modified * current_vertical_scale;

                for y in 0..CHUNK_SIZE {
                    let world_y = coords.1 as f64 * CHUNK_SIZE as f64 + y as f64;

                    let base_density = approx_world_height - world_y;

                    let density_nx = world_x * DENSITY_H_SCALE;
                    let density_ny = world_y * DENSITY_V_SCALE + 0.1;
                    let density_nz = world_z * DENSITY_H_SCALE - 0.1;
                    let density_noise_val = density_noise.get([density_nx, density_ny, density_nz]);

                    let mut final_density = base_density + density_noise_val * density_weight;

                    let floor_factor = (world_y / CAVE_FLOOR_DEPTH).clamp(0.0, 1.0);
                    final_density += (1.0 - floor_factor) * 10.0;

                    let mut is_cave_air = false;
                    if world_y > CAVE_MIN_Y
                        && world_y < CAVE_MAX_Y
                        && world_y < approx_world_height - CAVE_SURFACE_BUFFER
                    {
                        if !is_cave_air {
                            let cave_nx = world_x * CAVE_H_SCALE - 5.0;
                            let cave_ny = world_y * CAVE_V_SCALE - 10.0;
                            let cave_nz = world_z * CAVE_H_SCALE - 15.0;
                            let cave_val = cave_noise.get([cave_nx, cave_ny, cave_nz]);
                            if cave_val > CAVE_THRESHOLD {
                                is_cave_air = true;
                            }
                        }
                    }

                    if is_cave_air {
                        chunk.blocks[x][y][z] = None;
                    } else if final_density > DENSITY_THRESHOLD {
                        chunk.blocks[x][y][z] = Some(BlockType::Stone);
                    } else {
                        chunk.blocks[x][y][z] = None;
                    }
                    if world_y < 1.0 {
                        chunk.blocks[x][y][z] = Some(BlockType::Stone);
                        continue;
                    }
                }

                let dirt_depth = 3;

                for y in (0..CHUNK_SIZE).rev() {
                    let current_block = chunk.blocks[x][y][z];

                    let block_above = if y == CHUNK_SIZE - 1 {
                        None
                    } else {
                        chunk.blocks[x][y + 1][z]
                    };

                    if current_block == Some(BlockType::Stone) && block_above.is_none() {
                        chunk.blocks[x][y][z] = Some(BlockType::Grass);

                        for d in 1..=dirt_depth {
                            if y >= d {
                                let below_idx = y - d;
                                if chunk.blocks[x][below_idx][z] == Some(BlockType::Stone) {
                                    chunk.blocks[x][below_idx][z] = Some(BlockType::Dirt);
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }

                        break;
                    }
                }
            }
        }
        chunk
    }

    #[inline]
    pub fn get_local_block(&self, x: usize, y: usize, z: usize) -> Option<&BlockType> {
        self.blocks[x][y][z].as_ref()
    }
}

#[derive(Clone)]
pub struct ChunkNeighborhood {
    center_coords: ChunkCoords,
    center: Arc<ChunkBlocks>,
    north: Arc<ChunkBlocks>,
    south: Arc<ChunkBlocks>,
    east: Arc<ChunkBlocks>,
    west: Arc<ChunkBlocks>,
    up: Arc<ChunkBlocks>,
    down: Arc<ChunkBlocks>,
}

impl ChunkNeighborhood {
    #[inline]
    pub fn get_block(&self, gx: i32, gy: i32, gz: i32) -> Option<&BlockType> {
        let target_cx = gx.div_euclid(CHUNK_SIZE as i32);
        let target_cy = gy.div_euclid(CHUNK_SIZE as i32);
        let target_cz = gz.div_euclid(CHUNK_SIZE as i32);
        let target_coord = (target_cx, target_cy, target_cz);

        let lx = gx.rem_euclid(CHUNK_SIZE as i32) as usize;
        let ly = gy.rem_euclid(CHUNK_SIZE as i32) as usize;
        let lz = gz.rem_euclid(CHUNK_SIZE as i32) as usize;

        let dx = target_coord.0 - self.center_coords.0;
        let dy = target_coord.1 - self.center_coords.1;
        let dz = target_coord.2 - self.center_coords.2;

        let chunk_data = match (dx, dy, dz) {
            (0, 0, 0) => &self.center,
            (0, 0, 1) => &self.north,
            (0, 0, -1) => &self.south,
            (1, 0, 0) => &self.east,
            (-1, 0, 0) => &self.west,
            (0, 1, 0) => &self.up,
            (0, -1, 0) => &self.down,
            _ => return None,
        };

        chunk_data.get_local_block(lx, ly, lz)
    }

    pub fn get_center_coords(&self) -> ChunkCoords {
        self.center_coords
    }
}

pub struct World {
    chunks: HashMap<ChunkCoords, Arc<ChunkBlocks>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn insert_chunk_blocks(&mut self, coords: ChunkCoords, blocks: Arc<ChunkBlocks>) {
        self.chunks.insert(coords, blocks);
    }

    pub fn remove_chunk_blocks(&mut self, coords: &ChunkCoords) -> Option<Arc<ChunkBlocks>> {
        self.chunks.remove(coords)
    }

    pub fn chunk_exists(&self, coords: ChunkCoords) -> bool {
        self.chunks.contains_key(&coords)
    }

    pub fn _get_chunk_blocks(&self, coords: ChunkCoords) -> Option<Arc<ChunkBlocks>> {
        self.chunks.get(&coords).cloned()
    }

    /// Attempts to retrieve the block data for a central chunk and its four direct
    /// neighbors (+X, -X, +Z, -Z).
    ///
    /// Returns `Some(ChunkNeighborhood)` if the central chunk and all four neighbors
    /// exist in the world, otherwise returns `None`.
    ///
    /// This should be called *after* verifying that neighbors are ready for meshing.
    pub fn get_chunk_neighborhood(&self, center_coords: ChunkCoords) -> Option<ChunkNeighborhood> {
        let (cx, cy, cz) = center_coords;
        let north_coords = (cx, cy, cz + 1);
        let south_coords = (cx, cy, cz - 1);
        let east_coords = (cx + 1, cy, cz);
        let west_coords = (cx - 1, cy, cz);
        let up_coords = (cx, cy + 1, cz);
        let down_coords = (cx, cy - 1, cz);

        let center = self.chunks.get(&center_coords)?;
        let north = self.chunks.get(&north_coords)?;
        let south = self.chunks.get(&south_coords)?;
        let east = self.chunks.get(&east_coords)?;
        let west = self.chunks.get(&west_coords)?;
        let up = self.chunks.get(&up_coords)?;
        let down = self.chunks.get(&down_coords)?;

        Some(ChunkNeighborhood {
            center_coords,
            center: Arc::clone(center),
            north: Arc::clone(north),
            south: Arc::clone(south),
            east: Arc::clone(east),
            west: Arc::clone(west),
            up: Arc::clone(up),
            down: Arc::clone(down),
        })
    }
}
