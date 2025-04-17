use crate::worldgen::{BlockType, CHUNK_BREADTH, CHUNK_HEIGHT};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use std::collections::HashMap;
use std::sync::Arc;

pub type ChunkCoords = (i32, i32);
pub type LocalBlockCoords = (usize, usize, usize);

#[derive(Clone)]
pub struct ChunkBlocks {
    blocks: [[[Option<BlockType>; CHUNK_BREADTH]; CHUNK_HEIGHT]; CHUNK_BREADTH],
}

enum BiomeType {
    Plains,
    Hills,
    Mountains,
}

impl ChunkBlocks {
    fn new_empty() -> Self {
        Self {
            blocks: [[[None; CHUNK_BREADTH]; CHUNK_HEIGHT]; CHUNK_BREADTH],
        }
    }

    pub fn generate(coords: (i32, i32)) -> Self {
        let mut chunk = Self::new_empty();

        const SEED: u32 = 1;
        const SEA_LEVEL: f64 = (CHUNK_HEIGHT / 3) as f64;
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

        for x in 0..CHUNK_BREADTH {
            for z in 0..CHUNK_BREADTH {
                let world_x = coords.0 as f64 * CHUNK_BREADTH as f64 + x as f64;
                let world_z = coords.1 as f64 * CHUNK_BREADTH as f64 + z as f64;

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

                for y in 0..CHUNK_HEIGHT {
                    let world_y = y as f64;

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
                }

                let dirt_depth = 3;

                for y in (0..CHUNK_HEIGHT).rev() {
                    let current_block = chunk.blocks[x][y][z];

                    let block_above = if y == CHUNK_HEIGHT - 1 {
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

                if chunk.blocks[x][0][z].is_none()
                    || chunk.blocks[x][0][z] == Some(BlockType::Grass)
                    || chunk.blocks[x][0][z] == Some(BlockType::Dirt)
                {
                    chunk.blocks[x][0][z] = Some(BlockType::Stone);
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

    pub fn get_chunk_blocks(&self, coords: ChunkCoords) -> Option<Arc<ChunkBlocks>> {
        self.chunks.get(&coords).cloned()
    }

    #[inline]
    pub fn global_to_chunk_local(
        gx: i32,
        gy: i32,
        gz: i32,
    ) -> Option<(ChunkCoords, LocalBlockCoords)> {
        if !(0..CHUNK_HEIGHT as i32).contains(&gy) {
            return None;
        }

        let cx = gx.div_euclid(CHUNK_BREADTH as i32);
        let cz = gz.div_euclid(CHUNK_BREADTH as i32);

        let lx = gx.rem_euclid(CHUNK_BREADTH as i32) as usize;
        let ly = gy as usize;
        let lz = gz.rem_euclid(CHUNK_BREADTH as i32) as usize;

        Some(((cx, cz), (lx, ly, lz)))
    }

    #[inline]
    pub fn get_block(&self, gx: i32, gy: i32, gz: i32) -> Option<&BlockType> {
        if let Some((chunk_coords, local_coords)) = Self::global_to_chunk_local(gx, gy, gz) {
            if let Some(chunk_arc) = self.chunks.get(&chunk_coords) {
                chunk_arc.get_local_block(local_coords.0, local_coords.1, local_coords.2)
            } else {
                None
            }
        } else {
            None
        }
    }
}
