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

    fn generate(coords: ChunkCoords) -> Self {
        let mut chunk = Self::new_empty();

        let perlin = Perlin::new(1);
        for x in 0..CHUNK_BREADTH {
            for z in 0..CHUNK_BREADTH {
                let stone_height = CHUNK_HEIGHT / 3;

                let world_x = coords.0 as f64 * CHUNK_BREADTH as f64 + x as f64;
                let world_z = coords.1 as f64 * CHUNK_BREADTH as f64 + z as f64;
                let random =
                    (perlin.get([world_x * TERRAIN_SCALE, world_z * TERRAIN_SCALE]) + 1.0) * 0.5;
                let grass_height = ((CHUNK_HEIGHT - 1) as f64 * random) as usize;

                chunk.blocks[x][grass_height][z] = Some(BlockType::Grass);
                for y in 0..grass_height {
                    let block_type = if y < stone_height {
                        Some(BlockType::Stone)
                    } else {
                        Some(BlockType::Dirt)
                    };
                    chunk.blocks[x][y][z] = block_type;
                }
            }
        }
        chunk
    }

    #[inline]
    pub fn get_local_block(&self, x: usize, y: usize, z: usize) -> Option<&BlockType> {
        self.blocks[x][y][z].as_ref()
    }

    // // Potentially useful later: Set a block type at local coordinates
    // pub fn set_local_block(&mut self, x: usize, y: usize, z: usize, block_type: Option<BlockType>) {
    //     if x < CHUNK_BREADTH && y < CHUNK_HEIGHT && z < CHUNK_BREADTH {
    //         self.blocks[x][y][z] = block_type;
    //     }
    // }
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

    pub fn ensure_chunk_generated(&mut self, coords: ChunkCoords) {
        self.chunks
            .entry(coords)
            .or_insert_with(|| Arc::new(ChunkBlocks::generate(coords)));
    }

    pub fn get_chunk_blocks(&self, coords: ChunkCoords) -> Option<&Arc<ChunkBlocks>> {
        self.chunks.get(&coords)
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
            None // Y coordinate was out of bounds
        }
    }

    // // Potentially useful later: Set a block anywhere in the world
    // pub fn set_block(&mut self, gx: i32, gy: i32, gz: i32, block_type: Option<BlockType>) -> bool {
    //     if let Some((chunk_coords, local_coords)) = Self::global_to_chunk_local(gx, gy, gz) {
    //         // Ensure the chunk exists, generating if needed (or load from disk)
    //         let chunk_arc = self.chunks.entry(chunk_coords).or_insert_with(|| {
    //              println!("Generating block data for chunk {:?} due to set_block", chunk_coords);
    //             Arc::new(ChunkBlocks::generate(chunk_coords))
    //         });

    //         // Get a mutable reference to the ChunkBlocks inside the Arc.
    //         // This clones the ChunkBlocks if the Arc is shared (ref count > 1).
    //         let chunk_data = Arc::make_mut(chunk_arc);
    //         chunk_data.set_local_block(local_coords.0, local_coords.1, local_coords.2, block_type);
    //         true // Indicate success
    //     } else {
    //         false // Indicate out of bounds
    //     }
    // }
}
