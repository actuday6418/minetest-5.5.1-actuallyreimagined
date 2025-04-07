// ./src/model.rs
use glam::Vec3;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

/// Generates vertex positions, normals, and indices for multiple cubes.
/// Each cube is defined by 24 vertices (4 per face) to allow for correct face normals.
///
/// # Arguments
///
/// * `centers` - A slice of `Vec3` representing the center positions for each cube.
///
/// # Returns
///
/// A tuple containing:
/// * `Vec<Position>` - The combined vertex positions for all cubes.
/// * `Vec<Normal>` - The combined vertex normals for all cubes.
/// * `Vec<u16>` - The combined indices for all cubes.
pub fn generate_cube_mesh(centers: &[Vec3]) -> (Vec<Position>, Vec<Normal>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    // Define vertices relative to center (0,0,0) for a 1x1x1 cube
    // 24 vertices total (4 per face)
    const H: f32 = 0.5; // Half size
    let rel_vertices = [
        // Front face (+Z) Normal (0, 0, 1)
        Vec3::new(-H, -H, H),
        Vec3::new(H, -H, H),
        Vec3::new(H, H, H),
        Vec3::new(-H, H, H),
        // Back face (-Z) Normal (0, 0, -1)
        Vec3::new(-H, -H, -H),
        Vec3::new(-H, H, -H),
        Vec3::new(H, H, -H),
        Vec3::new(H, -H, -H),
        // Left face (-X) Normal (-1, 0, 0)
        Vec3::new(-H, -H, -H),
        Vec3::new(-H, -H, H),
        Vec3::new(-H, H, H),
        Vec3::new(-H, H, -H),
        // Right face (+X) Normal (1, 0, 0)
        Vec3::new(H, -H, -H),
        Vec3::new(H, H, -H),
        Vec3::new(H, H, H),
        Vec3::new(H, -H, H),
        // Top face (+Y) Normal (0, 1, 0)
        Vec3::new(-H, H, -H),
        Vec3::new(-H, H, H),
        Vec3::new(H, H, H),
        Vec3::new(H, H, -H),
        // Bottom face (-Y) Normal (0, -1, 0)
        Vec3::new(-H, -H, -H),
        Vec3::new(H, -H, -H),
        Vec3::new(H, -H, H),
        Vec3::new(-H, -H, H),
    ];

    let face_normals = [
        // Front
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        // Back
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        // Left
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        // Right
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        // Top
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        // Bottom
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];

    let base_indices: [u32; 36] = [
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];

    let mut current_vertex_offset: u32 = 0;
    for center in centers {
        for i in 0..24 {
            let world_pos = *center + rel_vertices[i];
            positions.push(Position {
                position: world_pos.to_array(),
            });
            normals.push(Normal {
                normal: face_normals[i].to_array(),
            });
        }
        for index in base_indices.iter() {
            indices.push(current_vertex_offset + index);
        }
        current_vertex_offset += 24;
    }

    (positions, normals, indices)
}
