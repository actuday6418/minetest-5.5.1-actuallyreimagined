use glam::IVec3;
use std::collections::HashMap;

pub mod data;
pub mod lod;

use crate::world::{BlockType, ChunkCoords, ChunkNeighborhood};
use data::{FaceData, get_block_face_quad_index};

fn chunk_coords_to_ivec3(coords: ChunkCoords) -> IVec3 {
    IVec3::new(coords.0, coords.1, coords.2)
}

pub const CHUNK_SIZE: usize = 64;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_SIZE_P: usize = CHUNK_SIZE + 2;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FaceDir {
    Up,
    Down,
    Left,
    Right,
    Forward,
    Back,
}

impl FaceDir {
    pub fn worldgen_face_index(&self) -> usize {
        match self {
            FaceDir::Back => 0,    // +Z
            FaceDir::Forward => 1, // -Z
            FaceDir::Left => 2,    // -X
            FaceDir::Right => 3,   // +X
            FaceDir::Up => 4,      // +Y
            FaceDir::Down => 5,    // -Y
        }
    }
}

pub fn build_chunk_mesh(neighborhood: &ChunkNeighborhood) -> Vec<FaceData> {
    let mut faces = Vec::new();
    let chunk = &neighborhood.center;
    let center_chunk_origin_gs =
        chunk_coords_to_ivec3(neighborhood.get_center_coords()) * CHUNK_SIZE_I32;
    let mut axis_cols: [[[u128; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3] =
        [[[0; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3];
    let mut col_face_masks: [[[u128; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 6] =
        [[[0; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 6];

    #[inline]
    fn add_voxel_to_axis_cols(
        b: BlockType,
        x: usize,
        y: usize,
        z: usize,
        axis_cols: &mut [[[u128; CHUNK_SIZE_P]; CHUNK_SIZE_P]; 3],
    ) {
        if b.is_opaque() {
            axis_cols[0][z][x] |= 1u128 << y as u128;
            axis_cols[1][y][z] |= 1u128 << x as u128;
            axis_cols[2][y][x] |= 1u128 << z as u128;
        }
    }

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                add_voxel_to_axis_cols(
                    chunk.get_local_block(x, y, z),
                    x + 1,
                    y + 1,
                    z + 1,
                    &mut axis_cols,
                )
            }
        }
    }

    for z in [0, CHUNK_SIZE_P - 1] {
        for y in 0..CHUNK_SIZE_P {
            for x in 0..CHUNK_SIZE_P {
                let pos = IVec3::new(x as i32, y as i32, z as i32)
                    + IVec3::ONE * center_chunk_origin_gs
                    - IVec3::ONE;
                add_voxel_to_axis_cols(
                    neighborhood
                        .get_block_from_global_coords(pos.x, pos.y, pos.z)
                        .unwrap_or_default(),
                    x,
                    y,
                    z,
                    &mut axis_cols,
                );
            }
        }
    }
    for z in 0..CHUNK_SIZE_P {
        for y in [0, CHUNK_SIZE_P - 1] {
            for x in 0..CHUNK_SIZE_P {
                let pos = IVec3::new(x as i32, y as i32, z as i32)
                    + IVec3::ONE * center_chunk_origin_gs
                    - IVec3::ONE;
                add_voxel_to_axis_cols(
                    neighborhood
                        .get_block_from_global_coords(pos.x, pos.y, pos.z)
                        .unwrap_or_default(),
                    x,
                    y,
                    z,
                    &mut axis_cols,
                );
            }
        }
    }
    for z in 0..CHUNK_SIZE_P {
        for x in [0, CHUNK_SIZE_P - 1] {
            for y in 0..CHUNK_SIZE_P {
                let pos = IVec3::new(x as i32, y as i32, z as i32)
                    + IVec3::ONE * center_chunk_origin_gs
                    - IVec3::ONE;
                add_voxel_to_axis_cols(
                    neighborhood
                        .get_block_from_global_coords(pos.x, pos.y, pos.z)
                        .unwrap_or_default(),
                    x,
                    y,
                    z,
                    &mut axis_cols,
                );
            }
        }
    }

    for axis_idx in 0..3 {
        for i1 in 0..CHUNK_SIZE_P {
            for i2 in 0..CHUNK_SIZE_P {
                let col = axis_cols[axis_idx][i1][i2];
                let pos_face_mask = col & !(col << 1);
                let neg_face_mask = col & !(col >> 1);
                match axis_idx {
                    0 => {
                        col_face_masks[0][i1][i2] = neg_face_mask;
                        col_face_masks[1][i1][i2] = pos_face_mask;
                    }
                    1 => {
                        col_face_masks[2][i1][i2] = neg_face_mask;
                        col_face_masks[3][i1][i2] = pos_face_mask;
                    }
                    2 => {
                        col_face_masks[4][i1][i2] = neg_face_mask;
                        col_face_masks[5][i1][i2] = pos_face_mask;
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    let mut greedy_data: [HashMap<u32, HashMap<u32, [u64; CHUNK_SIZE]>>; 6] = Default::default();

    for axis_enum_idx in 0..6 {
        let face_dir = match axis_enum_idx {
            0 => FaceDir::Down,
            1 => FaceDir::Up,
            2 => FaceDir::Left,
            3 => FaceDir::Right,
            4 => FaceDir::Forward,
            _ => FaceDir::Back,
        };
        for plane_v in 0..CHUNK_SIZE {
            for plane_u in 0..CHUNK_SIZE {
                let padded_u = plane_u + 1;
                let padded_v = plane_v + 1;
                let mut face_col_bits = match axis_enum_idx {
                    0 | 1 => col_face_masks[axis_enum_idx][padded_v][padded_u],
                    2 | 3 => col_face_masks[axis_enum_idx][padded_u][padded_v],
                    _ => col_face_masks[axis_enum_idx][padded_v][padded_u],
                };
                face_col_bits >>= 1;
                face_col_bits &= !(1 << CHUNK_SIZE as u128);
                while face_col_bits != 0 {
                    let slice_coord = face_col_bits.trailing_zeros();
                    face_col_bits &= face_col_bits - 1;

                    let voxel_pos_local = match face_dir {
                        FaceDir::Up => {
                            IVec3::new(plane_u as i32, slice_coord as i32, plane_v as i32)
                        }
                        FaceDir::Right => {
                            IVec3::new(slice_coord as i32, plane_u as i32, plane_v as i32)
                        }
                        FaceDir::Back => {
                            IVec3::new(plane_u as i32, plane_v as i32, slice_coord as i32)
                        }
                        FaceDir::Down => {
                            IVec3::new(plane_u as i32, slice_coord as i32, plane_v as i32)
                        }
                        FaceDir::Left => {
                            IVec3::new(slice_coord as i32, plane_u as i32, plane_v as i32)
                        }
                        FaceDir::Forward => {
                            IVec3::new(plane_u as i32, plane_v as i32, slice_coord as i32)
                        }
                    };
                    let block_type = neighborhood.center.get_local_block(
                        voxel_pos_local.x as usize,
                        voxel_pos_local.y as usize,
                        voxel_pos_local.z as usize,
                    );
                    let type_map = greedy_data[axis_enum_idx]
                        .entry(block_type as u32)
                        .or_default();
                    let plane_mask = type_map.entry(slice_coord).or_insert([0; 64]);
                    plane_mask[plane_u] |= 1u64 << (plane_v as u64);
                }
            }
        }
    }

    for (axis_enum_idx, type_data) in greedy_data.into_iter().enumerate() {
        let face_dir = match axis_enum_idx {
            0 => FaceDir::Down,
            1 => FaceDir::Up,
            2 => FaceDir::Left,
            3 => FaceDir::Right,
            4 => FaceDir::Forward,
            _ => FaceDir::Back,
        };

        for (block_type, slice_data) in type_data.into_iter() {
            for (slice_coord_u32, plane_mask) in slice_data.into_iter() {
                let quads = greedy_mesh_binary_plane(plane_mask, CHUNK_SIZE as u32);

                for q in quads {
                    let (origin_x, origin_y, origin_z) =
                        calculate_face_origin(face_dir, slice_coord_u32, &q);
                    let (u_scale, v_scale) = match face_dir {
                        FaceDir::Up | FaceDir::Down => (q.w, q.h),
                        FaceDir::Left | FaceDir::Right => (q.h, q.w),
                        FaceDir::Forward | FaceDir::Back => (q.w, q.h),
                    };
                    let u_scale_m1 = u_scale.saturating_sub(1).min(63);
                    let v_scale_m1 = v_scale.saturating_sub(1).min(63);
                    let packed_origin = origin_x | (origin_y << 8) | (origin_z << 16);
                    let worldgen_face_idx = face_dir.worldgen_face_index();
                    let quad_index = get_block_face_quad_index(
                        block_type.try_into().unwrap_or(BlockType::Air),
                        worldgen_face_idx,
                    );

                    let packed_scale_quad_index =
                        u_scale_m1 | (v_scale_m1 << 6) | (quad_index << 12);

                    faces.push(FaceData {
                        packed_origin,
                        packed_scale_quad_index,
                    });
                }
            }
        }
    }

    faces
}

#[inline]
fn calculate_face_origin(
    face_dir: FaceDir,
    slice_coord: u32,
    quad: &GreedyQuad,
) -> (u32, u32, u32) {
    let qx = quad.x;
    let qy = quad.y;
    let sc = slice_coord;

    match face_dir {
        FaceDir::Up => (qx, sc, qy),
        FaceDir::Down => (qx, sc + 1, qy),

        FaceDir::Right => (sc, qx, qy),
        FaceDir::Left => (sc + 1, qx, qy),

        FaceDir::Back => (qx, qy, sc),
        FaceDir::Forward => (qx, qy, sc + 1),
    }
}

#[derive(Debug)]
struct GreedyQuad {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

fn greedy_mesh_binary_plane(mut data: [u64; CHUNK_SIZE], plane_size: u32) -> Vec<GreedyQuad> {
    let mut greedy_quads = vec![];

    for x in 0..plane_size as usize {
        let mut current_col_bits = data[x];
        let mut y = 0;
        while y < plane_size {
            let skip = current_col_bits.trailing_zeros();
            if skip >= plane_size {
                break;
            }
            y += skip;
            current_col_bits >>= skip;
            let h = current_col_bits.trailing_ones();
            if y + h > plane_size {
                break;
            }
            let h_mask = u64::checked_shl(1, h).map_or(!0, |v| v - 1);
            let mut w = 1;
            while x + w < plane_size as usize {
                let next_col_segment = (data[x + w] >> y) & h_mask;
                if next_col_segment != h_mask {
                    break;
                }
                w += 1;
            }
            let clear_mask = !(h_mask << y);
            for i in 0..w {
                data[x + i] &= clear_mask;
            }
            greedy_quads.push(GreedyQuad {
                x: x as u32,
                y,
                w: w as u32,
                h,
            });
            current_col_bits >>= h;
            y += h;
        }
    }

    greedy_quads
}
