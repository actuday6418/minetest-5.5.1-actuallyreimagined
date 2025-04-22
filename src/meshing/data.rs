use crate::world::BlockType;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

const TEXTURE_SIZE_PIXELS: f32 = 18.0;
const TEXTURE_PADDING_PIXELS: f32 = 1.0;
pub const ATLAS_W: f32 = 1024.0;
pub const ATLAS_H: f32 = 1024.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct QuadTemplateData {
    pub model_positions: [[f32; 4]; 4],
    pub uvs: [[f32; 2]; 4],
    pub normal_index: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FaceData {
    // Bit structure:
    // 0-7:   Local X position (origin of quad)
    // 8-15:  Local Y position (origin of quad)
    // 16-23: Local Z position (origin of quad)
    // 24-31: Unused
    //
    // Note: The axis represented by U and V depends on the face normal.
    //  e.g., for +Z face: U=X, V=Y
    pub packed_origin: u32,
    // Instance data 2: Scale and Quad Index
    // Bits 0-5:   U scale - 1 (0-63) -> 6 bits
    // Bits 6-11:  V scale - 1 (0-63) -> 6 bits
    // Bits 12-31: Quad Index         -> 20 bits (Allows for ~1 million quad templates)
    pub packed_scale_quad_index: u32,
}

impl FaceData {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FaceData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

const STONE_FACE_QUADS_START: u32 = 0;
const DIRT_FACE_QUADS_START: u32 = 6;
const GRASS_FACE_QUADS_START: u32 = 12;

pub fn get_block_face_quad_index(block_type: BlockType, face_index: usize) -> u32 {
    let base = match block_type {
        BlockType::Stone => STONE_FACE_QUADS_START,
        BlockType::Dirt => DIRT_FACE_QUADS_START,
        BlockType::Grass => GRASS_FACE_QUADS_START,
        BlockType::Air => panic!("Water should not generate quads"),
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
        dirt_region,
        grass_top_region,
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

    let base_uvs = [
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 0.0),
    ];

    let create_template = |face_idx: usize, region: [f32; 4]| -> QuadTemplateData {
        let mut model_positions = [[0.0; 4]; 4];
        let mut uvs = [[0.0; 2]; 4];
        let normal_index = face_idx as u32;

        let region_min_u = region[0];
        let region_min_v = region[1];
        let region_width = region[2] - region_min_u;
        let region_height = region[3] - region_min_v;

        for i in 0..4 {
            let vertex_index_in_cube_data = face_idx * 4 + i;
            let pos_vec3 = rel_vertices[vertex_index_in_cube_data];
            model_positions[i] = [pos_vec3.x, pos_vec3.y, pos_vec3.z, 0.0];

            let base_uv = base_uvs[i];
            uvs[i] = [
                region_min_u + base_uv.x * region_width,
                region_min_v + base_uv.y * region_height,
            ];
        }

        QuadTemplateData {
            model_positions,
            uvs,
            normal_index,
            _padding: [0, 0, 0],
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
