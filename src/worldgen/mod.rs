use glam::{Vec2, Vec3};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

const TEXTURE_SIZE_PIXELS: f32 = 16f32;
const ATLAS_W: f32 = 1024f32;
const ATLAS_H: f32 = 1024f32;
pub const CHUNK_BREADTH: i32 = 16;
pub const CHUNK_HEIGHT: i32 = 5;

const VERTEX_UVS: [Vec2; 24] = [
    // Front face (+Z)
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    // Back face (-Z)
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0),
    // Left face (-X)
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    // Right face (+X)
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    // Top face (+Y)
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    // Bottom face (-Y)
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

#[derive(PartialEq)]
pub enum BlockType {
    Water,
    Dirt,
    Grass,
    Stone,
}

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

fn get_uv_region(tex_x: f32, tex_y: f32) -> [f32; 4] {
    let min_u = tex_x / ATLAS_W;
    let min_v = tex_y / ATLAS_H;
    let max_u = (tex_x + TEXTURE_SIZE_PIXELS) / ATLAS_W;
    let max_v = (tex_y + TEXTURE_SIZE_PIXELS) / ATLAS_H;

    [min_u, min_v, max_u, max_v]
}

pub fn generate_chunk(start_pos: Vec3) -> (Vec<Position>, Vec<Normal>, Vec<TexCoord>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut tex_coords = Vec::new();
    let mut indices = Vec::new();

    let stone_tex_coords = (TEXTURE_SIZE_PIXELS * 3.0, TEXTURE_SIZE_PIXELS * 0.0);
    let dirt_tex_coords = (TEXTURE_SIZE_PIXELS * 2.0, TEXTURE_SIZE_PIXELS * 0.0);
    let grass_top_tex_coords = (TEXTURE_SIZE_PIXELS * 1.0, TEXTURE_SIZE_PIXELS * 0.0);
    let grass_side_tex_coords = (TEXTURE_SIZE_PIXELS * 0.0, TEXTURE_SIZE_PIXELS * 0.0);

    let stone_region = get_uv_region(stone_tex_coords.0, stone_tex_coords.1);
    let dirt_region = get_uv_region(dirt_tex_coords.0, dirt_tex_coords.1);
    let grass_top_region = get_uv_region(grass_top_tex_coords.0, grass_top_tex_coords.1);
    let grass_side_region = get_uv_region(grass_side_tex_coords.0, grass_side_tex_coords.1);

    let stone_face_regions = [stone_region; 6];
    let dirt_face_regions = [dirt_region; 6];
    let grass_face_regions = [
        grass_side_region,
        grass_side_region,
        grass_side_region,
        grass_side_region,
        grass_top_region,
        dirt_region,
    ];

    const H: f32 = 0.5;
    let rel_vertices = [
        Vec3::new(-H, -H, H),
        Vec3::new(H, -H, H),
        Vec3::new(H, H, H),
        Vec3::new(-H, H, H), // Front 0 (+Z)
        Vec3::new(-H, -H, -H),
        Vec3::new(-H, H, -H),
        Vec3::new(H, H, -H),
        Vec3::new(H, -H, -H), // Back 1 (-Z)
        Vec3::new(-H, -H, -H),
        Vec3::new(-H, -H, H),
        Vec3::new(-H, H, H),
        Vec3::new(-H, H, -H), // Left 2 (-X)
        Vec3::new(H, -H, -H),
        Vec3::new(H, H, -H),
        Vec3::new(H, H, H),
        Vec3::new(H, -H, H), // Right 3 (+X)
        Vec3::new(-H, H, -H),
        Vec3::new(-H, H, H),
        Vec3::new(H, H, H),
        Vec3::new(H, H, -H), // Top 4 (+Y)
        Vec3::new(-H, -H, -H),
        Vec3::new(H, -H, -H),
        Vec3::new(H, -H, H),
        Vec3::new(-H, -H, H), // Bottom 5 (-Y)
    ];

    let face_normals = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];

    let face_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
    let neighbor_offsets: [(i32, i32, i32); 6] = [
        (0, 0, 1),
        (0, 0, -1),
        (-1, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    ];

    let mut current_vertex_offset: u32 = 0;

    for x in 0..CHUNK_BREADTH {
        for z in 0..CHUNK_BREADTH {
            for y in 0..CHUNK_HEIGHT {
                let block_type = match y {
                    0 | 1 => BlockType::Stone,
                    2 | 3 => BlockType::Dirt,
                    4 => BlockType::Grass,
                    _ => unreachable!(),
                };

                let center = start_pos + Vec3::new(x as f32 + H, y as f32 + H, z as f32 + H);

                let block_face_regions = match block_type {
                    BlockType::Stone => &stone_face_regions,
                    BlockType::Dirt => &dirt_face_regions,
                    BlockType::Grass => &grass_face_regions,
                    _ => continue,
                };

                for face_index in 0..6 {
                    let offset = neighbor_offsets[face_index];
                    let neighbor_x = x + offset.0;
                    let neighbor_y = y + offset.1;
                    let neighbor_z = z + offset.2;

                    let neighbor_is_solid = neighbor_x >= 0
                        && neighbor_x < CHUNK_BREADTH
                        && neighbor_y >= 0
                        && neighbor_y < CHUNK_HEIGHT
                        && neighbor_z >= 0
                        && neighbor_z < CHUNK_BREADTH;

                    if !neighbor_is_solid {
                        let vertex_start_index_in_cube_data = face_index * 4;
                        let normal = face_normals[face_index];
                        let region = block_face_regions[face_index];

                        let region_min_u = region[0];
                        let region_min_v = region[1];
                        let region_max_u = region[2];
                        let region_max_v = region[3];
                        let region_width = region_max_u - region_min_u;
                        let region_height = region_max_v - region_min_v;

                        for i in 0..4 {
                            let data_idx = vertex_start_index_in_cube_data + i;
                            let world_pos = center + rel_vertices[data_idx];
                            positions.push(Position {
                                position: world_pos.to_array(),
                            });
                            normals.push(Normal {
                                normal: normal.to_array(),
                            });

                            let relative_uv = VERTEX_UVS[data_idx];
                            let final_u = region_min_u + relative_uv.x * region_width;
                            let final_v = region_min_v + relative_uv.y * region_height;
                            tex_coords.push(TexCoord {
                                tex_coord: [final_u, final_v],
                            });
                        }

                        for idx in face_indices.iter() {
                            indices.push(current_vertex_offset + idx);
                        }
                        current_vertex_offset += 4;
                    }
                }
            }
        }
    }

    (positions, normals, tex_coords, indices)
}
