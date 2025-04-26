use crate::aabb::AABB;
use glam::IVec3;
use std::collections::HashMap;

/// Trait for providing collision geometry from the game world.
///
/// The physics simulation uses this trait to query for potential colliders
/// within a specific region (the query_aabb).
pub trait PhysicsWorldProvider {
    /// Query the world for static collider AABBs that intersect `query_aabb`.
    fn query_potential_colliders(&self, query_aabb: AABB) -> Vec<AABB>;
}

/// Represents a type of block in the world.
pub trait HasAABB {
    /// Gets the collision AABB for this block type if it's solid.
    /// The AABB should be relative to the block's origin (min corner at 0,0,0).
    /// Returns `None` if the block type is non-collidable (like air).
    fn get_relative_aabb(&self) -> Option<AABB>;

    // Helper method to get world AABB, translating the relative one.
    // Not strictly required by the trait, but useful.
    fn get_world_aabb(&self, block_pos_world: IVec3) -> Option<AABB> {
        self.get_relative_aabb()
            .map(|rel_aabb| rel_aabb.translate(block_pos_world.as_vec3()))
    }
}

#[derive(Clone)]
pub struct ChunkedWorld<B, const CHUNK_SIZE: usize>
where
    B: HasAABB + Clone, // BlockType must be Cloneable to fill chunks
{
    chunks: HashMap<IVec3, Box<[[[B; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]>>,
}

impl<B, const CHUNK_SIZE: usize> ChunkedWorld<B, CHUNK_SIZE>
where
    B: HasAABB + Clone,
{
    /// Creates a new, empty `ChunkedWorld`.
    pub fn new() -> Self {
        const { assert!(CHUNK_SIZE > 0, "CHUNK_SIZE must be positive") }
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn load_chunk(
        &mut self,
        chunk_coord: IVec3,
        block_data: Box<[[[B; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]>,
    ) {
        self.chunks.insert(chunk_coord, block_data);
    }

    pub fn unload_chunk(&mut self, chunk_coord: IVec3) {
        self.chunks.remove(&chunk_coord);
    }

    pub fn get_block(&self, world_pos: IVec3) -> Option<&B> {
        let chunk_coord = IVec3::new(
            world_pos.x.div_euclid(CHUNK_SIZE as i32),
            world_pos.y.div_euclid(CHUNK_SIZE as i32),
            world_pos.z.div_euclid(CHUNK_SIZE as i32),
        );
        let local_coord = IVec3::new(
            world_pos.x.rem_euclid(CHUNK_SIZE as i32),
            world_pos.y.rem_euclid(CHUNK_SIZE as i32),
            world_pos.z.rem_euclid(CHUNK_SIZE as i32),
        );

        self.chunks.get(&chunk_coord).map(|chunk_data| {
            &chunk_data[local_coord.x as usize][local_coord.y as usize][local_coord.z as usize]
        })
    }
}

impl<B, const CHUNK_SIZE: usize> Default for ChunkedWorld<B, CHUNK_SIZE>
where
    B: HasAABB + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, const CHUNK_SIZE: usize> PhysicsWorldProvider for ChunkedWorld<B, CHUNK_SIZE>
where
    B: HasAABB + Clone,
{
    fn query_potential_colliders(&self, query_aabb: AABB) -> Vec<AABB> {
        let mut colliders = Vec::new();

        let min_block = query_aabb.min.floor().as_ivec3();
        let max_block = query_aabb.max.floor().as_ivec3();

        for x in min_block.x..=max_block.x {
            for y in min_block.y..=max_block.y {
                for z in min_block.z..=max_block.z {
                    let block_world_pos = IVec3::new(x, y, z);

                    let chunk_coord = IVec3::new(
                        x.div_euclid(CHUNK_SIZE as i32),
                        y.div_euclid(CHUNK_SIZE as i32),
                        z.div_euclid(CHUNK_SIZE as i32),
                    );
                    let local_coord = IVec3::new(
                        x.rem_euclid(CHUNK_SIZE as i32),
                        y.rem_euclid(CHUNK_SIZE as i32),
                        z.rem_euclid(CHUNK_SIZE as i32),
                    );

                    if let Some(chunk_data) = self.chunks.get(&chunk_coord) {
                        let block_type = &chunk_data[local_coord.x as usize]
                            [local_coord.y as usize][local_coord.z as usize];

                        if let Some(aabb) = block_type.get_world_aabb(block_world_pos) {
                            if query_aabb.intersects(&aabb) {
                                colliders.push(aabb);
                            }
                        }
                    }
                }
            }
        }

        colliders
    }
}
