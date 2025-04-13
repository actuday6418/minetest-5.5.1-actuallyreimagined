use crate::world::{ChunkCoords, World};
use glam::{Vec2, Vec3};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

const TEXTURE_SIZE_PIXELS: f32 = 18.0;
const TEXTURE_PADDING_PIXELS: f32 = 1.0;
pub const ATLAS_W: f32 = 1024.0;
pub const ATLAS_H: f32 = 1024.0;
pub const CHUNK_BREADTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 250;

#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct QuadTemplateData {
    pub model_positions: [[f32; 4]; 4], // relative to block center. Ignore extra f32 for padding.
    pub uvs: [[f32; 2]; 4],
    pub normal: [f32; 4], // ignore extra f32 for padding
}

#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub struct FaceData {
    #[format(R8G8B8_UINT)]
    pub block_position: [u8; 3],
    #[format(R32_UINT)]
    pub quad_index: u32,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum BlockType {
    _Water,
    Dirt,
    Grass,
    Stone,
}

impl BlockType {
    pub fn is_opaque(&self) -> bool {
        match self {
            BlockType::_Water => false,
            _ => true,
        }
    }
}
const STONE_FACE_QUADS_START: u32 = 0;
const DIRT_FACE_QUADS_START: u32 = 6;
const GRASS_FACE_QUADS_START: u32 = 12;

fn get_block_face_quad_index(block_type: BlockType, face_index: usize) -> u32 {
    let base = match block_type {
        BlockType::Stone => STONE_FACE_QUADS_START,
        BlockType::Dirt => DIRT_FACE_QUADS_START,
        BlockType::Grass => GRASS_FACE_QUADS_START,
        BlockType::_Water => panic!("Water should not generate quads"),
    };
    base + face_index as u32
}

pub fn create_quad_templates() -> Vec<QuadTemplateData> {
    let mut templates = Vec::new();

    let stone_tex_coords = (
        TEXTURE_SIZE_PIXELS * 3.0 + 1.0,
        TEXTURE_SIZE_PIXELS * 0.0 + 1.0,
    );
    let dirt_tex_coords = (
        TEXTURE_SIZE_PIXELS * 2.0 + 1.0,
        TEXTURE_SIZE_PIXELS * 0.0 + 1.0,
    );
    let grass_top_tex_coords = (
        TEXTURE_SIZE_PIXELS * 1.0 + 1.0,
        TEXTURE_SIZE_PIXELS * 0.0 + 1.0,
    );
    let grass_side_tex_coords = (
        TEXTURE_SIZE_PIXELS * 0.0 + 1.0,
        TEXTURE_SIZE_PIXELS * 0.0 + 1.0,
    );

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
        // Front (+Z)
        Vec3::new(-H, -H, H),
        Vec3::new(H, -H, H),
        Vec3::new(H, H, H),
        Vec3::new(-H, H, H),
        // Back (-Z)
        Vec3::new(H, -H, -H),
        Vec3::new(-H, -H, -H),
        Vec3::new(-H, H, -H),
        Vec3::new(H, H, -H),
        // Left (-X)
        Vec3::new(-H, -H, -H),
        Vec3::new(-H, -H, H),
        Vec3::new(-H, H, H),
        Vec3::new(-H, H, -H),
        // Right (+X)
        Vec3::new(H, -H, H),
        Vec3::new(H, -H, -H),
        Vec3::new(H, H, -H),
        Vec3::new(H, H, H),
        // Top (+Y)
        Vec3::new(-H, H, H),
        Vec3::new(H, H, H),
        Vec3::new(H, H, -H),
        Vec3::new(-H, H, -H),
        // Bottom (-Y)
        Vec3::new(-H, -H, -H),
        Vec3::new(H, -H, -H),
        Vec3::new(H, -H, H),
        Vec3::new(-H, -H, H),
    ];

    let face_normals = [
        Vec3::new(0.0, 0.0, 1.0),  // Front +Z
        Vec3::new(0.0, 0.0, -1.0), // Back -Z
        Vec3::new(-1.0, 0.0, 0.0), // Left -X
        Vec3::new(1.0, 0.0, 0.0),  // Right +X
        Vec3::new(0.0, 1.0, 0.0),  // Top +Y
        Vec3::new(0.0, -1.0, 0.0), // Bottom -Y
    ];

    // (Bottom-Left, Bottom-Right, Top-Right, Top-Left)
    let base_uvs = [
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
    ];

    let create_template = |face_idx: usize, region: [f32; 4]| -> QuadTemplateData {
        let mut model_positions = [[0.0; 4]; 4];
        let mut uvs = [[0.0; 2]; 4];
        let [x, y, z] = face_normals[face_idx].to_array();
        let normal = [x, y, z, 0.0];

        let region_min_u = region[0];
        let region_min_v = region[1];
        let region_width = region[2] - region_min_u;
        let region_height = region[3] - region_min_v;

        for i in 0..4 {
            let vertex_index_in_cube_data = face_idx * 4 + i;
            model_positions[i][0..3]
                .copy_from_slice(&rel_vertices[vertex_index_in_cube_data].to_array());

            let base_uv = base_uvs[i];
            uvs[i] = [
                region_min_u + base_uv.x * region_width,
                region_min_v + base_uv.y * region_height,
            ];
        }

        QuadTemplateData {
            model_positions,
            uvs,
            normal,
        }
    };

    let block_types_and_regions = [
        (BlockType::Stone, &stone_face_regions),
        (BlockType::Dirt, &dirt_face_regions),
        (BlockType::Grass, &grass_face_regions),
    ];

    for (_block_type, regions) in block_types_and_regions.iter() {
        for face_idx in 0..6 {
            templates.push(create_template(face_idx, regions[face_idx]));
        }
    }

    templates
}

fn get_uv_region(tex_x: f32, tex_y: f32) -> [f32; 4] {
    let min_u = tex_x / ATLAS_W;
    let min_v = tex_y / ATLAS_H;
    let max_u = (tex_x + TEXTURE_SIZE_PIXELS - 2.0 * TEXTURE_PADDING_PIXELS) / ATLAS_W;
    let max_v = (tex_y + TEXTURE_SIZE_PIXELS - 2.0 * TEXTURE_PADDING_PIXELS) / ATLAS_H;

    [min_u, min_v, max_u, max_v]
}

pub fn generate_chunk_mesh(chunk_coords: ChunkCoords, world: &World) -> Vec<FaceData> {
    let chunk_data = world.get_chunk_blocks(chunk_coords).unwrap();

    let mut faces = Vec::new();

    let neighbor_offsets: [(i32, i32, i32); 6] = [
        (0, 0, 1),
        (0, 0, -1),
        (-1, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    ];
    let chunk_origin_x = chunk_coords.0 * CHUNK_BREADTH as i32;
    let chunk_origin_z = chunk_coords.1 * CHUNK_BREADTH as i32;

    for lx in 0..CHUNK_BREADTH {
        for ly in 0..CHUNK_HEIGHT {
            for lz in 0..CHUNK_BREADTH {
                let block_type = match chunk_data.get_local_block(lx, ly, lz) {
                    Some(bt) => bt,
                    None => continue,
                };

                if !block_type.is_opaque() {
                    continue;
                }

                let gx = chunk_origin_x + lx as i32;
                let gy = ly as i32;
                let gz = chunk_origin_z + lz as i32;

                for face_index in 0..6 {
                    let offset = neighbor_offsets[face_index];
                    let neighbor_gx = gx + offset.0;
                    let neighbor_gy = gy + offset.1;
                    let neighbor_gz = gz + offset.2;

                    let neighbor_block = world.get_block(neighbor_gx, neighbor_gy, neighbor_gz);

                    let neighbor_is_opaque =
                        neighbor_block.map_or(false, |neighbor_type| neighbor_type.is_opaque());

                    if !neighbor_is_opaque {
                        let quad_index = get_block_face_quad_index(*block_type, face_index);
                        faces.push(FaceData {
                            block_position: [lx as u8, ly as u8, lz as u8],
                            quad_index,
                        });
                    }
                }
            }
        }
    }

    faces
}
