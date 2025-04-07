// ./src/model.rs
use glam::{Vec2, Vec3};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

const VERTEX_UVS: [Vec2; 24] = [
    // Front face (+Z) - Vertices 0-3 (BL, BR, TR, TL)
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    // Back face (-Z) - Vertices 4-7 (BL, TL, TR, BR relative to back view) -> Needs UVs (BL, TL, TR, BR)
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0), // Adjusted for back view: map vertices to Tex BL,TL,TR,BR
    // Left face (-X) - Vertices 8-11 (Bottom-Back, Bottom-Front, Top-Front, Top-Back relative to left view) -> Needs UVs (BR, BL, TL, TR)
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0), // Adjusted for left view
    // Right face (+X) - Vertices 12-15 (Bottom-Back, Top-Back, Top-Front, Bottom-Front relative to right view) -> Needs UVs (BL, TL, TR, BR)
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0), // Adjusted for right view
    // Top face (+Y) - Vertices 16-19 (Back-Left, Front-Left, Front-Right, Back-Right relative to top view) -> Needs UVs (BL, TL, TR, BR)
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0), // Adjusted for top view
    // Bottom face (-Y) - Vertices 20-23 (Back-Left, Back-Right, Front-Right, Front-Left relative to bottom view) -> Needs UVs (TL, TR, BR, BL)
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0), // Adjusted for bottom view
];

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

#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub struct TexCoord {
    #[format(R32G32_SFLOAT)]
    tex_coord: [f32; 2],
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
pub fn generate_cube_mesh(
    centers: &[Vec3],
    atlas_w: u32,
    atlas_h: u32,
) -> (Vec<Position>, Vec<Normal>, Vec<TexCoord>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut tex_coords = Vec::new();
    let mut indices = Vec::new();
    let atlas_w = atlas_w as f32;
    let atlas_h = atlas_h as f32;

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

    let tex_size_pixels = 16.0;
    let side_region = [
        tex_size_pixels / atlas_w,
        tex_size_pixels / atlas_h,
        0.0 / atlas_w,
        0.0 / atlas_h,
    ];
    let top_region = [
        tex_size_pixels * 2.0 / atlas_w,
        0.0 / atlas_h,
        tex_size_pixels * 3.0 / atlas_w,
        tex_size_pixels / atlas_h,
    ];
    let bottom_region = [
        tex_size_pixels / atlas_w,
        0.0 / atlas_h,
        (tex_size_pixels * 2.0) / atlas_w,
        tex_size_pixels / atlas_h,
    ];

    let face_regions = [
        side_region,
        side_region,
        side_region,
        side_region,
        top_region,
        bottom_region,
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
            let face_index = i / 4;

            let relative_uv = VERTEX_UVS[i];
            let region = face_regions[face_index];
            let region_min_u = region[0];
            let region_min_v = region[1];
            let region_width = region[2] - region_min_u;
            let region_height = region[3] - region_min_v;
            let final_u = region_min_u + relative_uv.x * region_width;
            let final_v = region_min_v + relative_uv.y * region_height;
            tex_coords.push(TexCoord {
                tex_coord: [final_u, final_v],
            });
        }
        for index in base_indices.iter() {
            indices.push(current_vertex_offset + index);
        }
        current_vertex_offset += 24;
    }

    (positions, normals, tex_coords, indices)
}
