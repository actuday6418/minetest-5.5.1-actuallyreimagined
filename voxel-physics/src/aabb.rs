use glam::Vec3;

/// An Axis-Aligned Bounding Box.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(a: Vec3, b: Vec3) -> Self {
        AABB {
            min: a.min(b),
            max: a.max(b),
        }
    }

    pub fn from_center_dims(center: Vec3, dimensions: Vec3) -> Self {
        let half_dims = dimensions * 0.5;
        AABB {
            min: center - half_dims,
            max: center + half_dims,
        }
    }

    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        self.min + self.size() * 0.5
    }

    #[inline]
    pub fn translate(&self, translation: Vec3) -> Self {
        AABB {
            min: self.min + translation,
            max: self.max + translation,
        }
    }

    #[inline]
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
            && self.min.z < other.max.z
            && self.max.z > other.min.z
    }

    /// Smallest AABB containing both.
    #[inline]
    pub fn union(&self, other: &AABB) -> Self {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn expanded(&self, amount: Vec3) -> Self {
        AABB {
            min: self.min - amount,
            max: self.max + amount,
        }
    }
}
