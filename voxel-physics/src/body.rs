use crate::aabb::AABB;
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct PhysicsBody {
    /// center of the AABB.
    pub position: Vec3,
    pub velocity: Vec3,
    pub dimensions: Vec3,
    pub is_grounded: bool,
}

impl PhysicsBody {
    pub fn new(position: Vec3, dimensions: Vec3) -> Self {
        PhysicsBody {
            position,
            velocity: Vec3::ZERO,
            dimensions,
            is_grounded: false,
        }
    }

    /// World-space AABB of body at current position.
    pub fn get_world_aabb(&self) -> AABB {
        AABB::from_center_dims(self.position, self.dimensions)
    }

    /// World-space AABB of body if it were at a specific position.
    pub fn get_world_aabb_at(&self, position: Vec3) -> AABB {
        AABB::from_center_dims(position, self.dimensions)
    }
}
