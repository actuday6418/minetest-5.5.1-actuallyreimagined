#![allow(dead_code)]

#[derive(Copy, Clone)]
pub enum Lod {
    L64,
    L32,
    L16,
    L8,
    L4,
    L2,
}

impl Lod {
    pub fn size(&self) -> i32 {
        match self {
            Lod::L64 => 64,
            Lod::L32 => 32,
            Lod::L16 => 16,
            Lod::L8 => 8,
            Lod::L4 => 4,
            Lod::L2 => 2,
        }
    }

    pub fn jump_index(&self) -> i32 {
        match self {
            Lod::L64 => 1,
            Lod::L32 => 2,
            Lod::L16 => 4,
            Lod::L8 => 8,
            Lod::L4 => 16,
            Lod::L2 => 32,
        }
    }
}
