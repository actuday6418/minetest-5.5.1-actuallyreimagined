use crate::worldgen::{BlockType, CHUNK_BREADTH, CHUNK_HEIGHT};
use noise::{NoiseFn, Perlin};
use std::collections::HashMap;
use std::sync::Arc;

pub type ChunkCoords = (i32, i32);
pub type LocalBlockCoords = (usize, usize, usize);
const TERRAIN_SCALE: f64 = 0.04;

#[derive(Clone)]
pub struct ChunkBlocks {
    blocks: [[[Option<BlockType>; CHUNK_BREADTH]; CHUNK_HEIGHT]; CHUNK_BREADTH],
}

impl ChunkBlocks {
    fn new_empty() -> Self {
        Self {
            blocks: [[[None; CHUNK_BREADTH]; CHUNK_HEIGHT]; CHUNK_BREADTH],
        }
    }

    pub fn generate(coords: ChunkCoords) -> Self {
        let mut chunk = Self::new_empty();
        let stone_height = CHUNK_HEIGHT / 3;
        let perlin = Perlin::new(1);
        for x in 0..CHUNK_BREADTH {
            for z in 0..CHUNK_BREADTH {
                let world_x = coords.0 as f64 * CHUNK_BREADTH as f64 + x as f64;
                let world_z = coords.1 as f64 * CHUNK_BREADTH as f64 + z as f64;

                let noise_val = perlin.get([world_x * TERRAIN_SCALE, world_z * TERRAIN_SCALE]);
                let normalized_noise = (noise_val + 1.0) * 0.5;
                let height_range = (CHUNK_HEIGHT - 1 - stone_height) as f64;
                let terrain_height =
                    (stone_height as f64 + normalized_noise * height_range).round() as usize;

                for y in 0..=terrain_height {
                    if y == terrain_height {
                        chunk.blocks[x][y][z] = Some(BlockType::Grass);
                    } else if y >= stone_height {
                        chunk.blocks[x][y][z] = Some(BlockType::Dirt);
                    } else {
                        chunk.blocks[x][y][z] = Some(BlockType::Stone);
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

    pub fn ensure_chunk_generated(&mut self, coords: ChunkCoords) -> bool {
        if !self.chunks.contains_key(&coords) {
            let generated_blocks = Arc::new(ChunkBlocks::generate(coords));
            self.chunks.insert(coords, generated_blocks);
            true
        } else {
            false
        }
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
