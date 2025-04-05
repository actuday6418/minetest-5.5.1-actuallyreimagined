use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

pub const POSITIONS: [Position; 8] = [
    // Back face (Z = -40)
    Position {
        position: [-40.0, -40.0, -40.0],
    }, // 0: Bottom-Left-Back
    Position {
        position: [40.0, -40.0, -40.0],
    }, // 1: Bottom-Right-Back
    Position {
        position: [-40.0, 40.0, -40.0],
    }, // 2: Top-Left-Back
    Position {
        position: [40.0, 40.0, -40.0],
    }, // 3: Top-Right-Back
    // Front face (Z = 40)
    Position {
        position: [-40.0, -40.0, 40.0],
    }, // 4: Bottom-Left-Front
    Position {
        position: [40.0, -40.0, 40.0],
    }, // 5: Bottom-Right-Front
    Position {
        position: [-40.0, 40.0, 40.0],
    }, // 6: Top-Left-Front
    Position {
        position: [40.0, 40.0, 40.0],
    }, // 7: Top-Right-Front
];

#[derive(BufferContents, Vertex, Copy, Clone)]
#[repr(C)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

pub const NORMALS: [Normal; 8] = [
    Normal { normal: [0.0, 0.0, 0.0] }; 8 // Use array repetition syntax
];

pub const INDICES: [u16; 36] = [
    // Front face (+Z)
    4, 5, 6, // Bottom-Left-Front, Bottom-Right-Front, Top-Left-Front
    6, 5, 7, // Top-Left-Front, Bottom-Right-Front, Top-Right-Front
    // Back face (-Z)
    1, 0, 3, // Bottom-Right-Back, Bottom-Left-Back, Top-Right-Back
    3, 0, 2, // Top-Right-Back, Bottom-Left-Back, Top-Left-Back
    // Left face (-X)
    0, 4, 2, // Bottom-Left-Back, Bottom-Left-Front, Top-Left-Back
    2, 4, 6, // Top-Left-Back, Bottom-Left-Front, Top-Left-Front
    // Right face (+X)
    5, 1, 7, // Bottom-Right-Front, Bottom-Right-Back, Top-Right-Front
    7, 1, 3, // Top-Right-Front, Bottom-Right-Back, Top-Right-Back
    // Top face (+Y)
    6, 7, 2, // Top-Left-Front, Top-Right-Front, Top-Left-Back
    2, 7, 3, // Top-Left-Back, Top-Right-Front, Top-Right-Back
    // Bottom face (-Y)
    0, 1, 4, // Bottom-Left-Back, Bottom-Right-Back, Bottom-Left-Front
    4, 1, 5, // Bottom-Left-Front, Bottom-Right-Back, Bottom-Right-Front
];
