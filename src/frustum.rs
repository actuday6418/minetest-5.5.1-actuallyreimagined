use glam::Vec4Swizzles;
use glam::{Mat4, Vec3, Vec4};

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Plane {
    pub fn new(coeffs: Vec4) -> Self {
        let len = coeffs.xyz().length();
        if len == 0.0 {
            Self {
                normal: Vec3::ZERO,
                distance: 0.0,
            }
        } else {
            Self {
                normal: coeffs.xyz() / len,
                distance: coeffs.w / len,
            }
        }
    }

    #[inline]
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub fn is_behind_or_intersecting_plane(&self, plane: &Plane) -> bool {
        let n_vertex = Vec3::new(
            if plane.normal.x > 0.0 {
                self.max.x
            } else {
                self.min.x
            },
            if plane.normal.y > 0.0 {
                self.max.y
            } else {
                self.min.y
            },
            if plane.normal.z > 0.0 {
                self.max.z
            } else {
                self.min.z
            },
        );

        plane.distance_to_point(n_vertex) >= 0.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    pub fn from_view_proj(matrix: &Mat4) -> Self {
        let row0 = matrix.row(0);
        let row1 = matrix.row(1);
        let row2 = matrix.row(2);
        let row3 = matrix.row(3);

        let planes = [
            Plane::new(row3 + row0),
            Plane::new(row3 - row0),
            Plane::new(row3 + row1),
            Plane::new(row3 - row1),
            Plane::new(row2),
            Plane::new(row3 - row2),
        ];
        Self { planes }
    }

    #[inline]
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            if !aabb.is_behind_or_intersecting_plane(plane) {
                return false;
            }
        }
        true
    }
}
