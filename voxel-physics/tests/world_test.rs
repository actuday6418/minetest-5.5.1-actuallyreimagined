use glam::{IVec3, Vec3};
use voxel_physics::{AABB, ChunkedWorld, HasAABB, PhysicsWorldProvider};

const TEST_CHUNK_SIZE: usize = 16; // Example chunk size

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TestBlock {
    Air,
    Solid,
}

impl HasAABB for TestBlock {
    fn get_relative_aabb(&self) -> Option<AABB> {
        match self {
            TestBlock::Air => None,
            TestBlock::Solid => Some(AABB::new(Vec3::ZERO, Vec3::ONE)),
        }
    }
}

type TestWorld = ChunkedWorld<TestBlock, TEST_CHUNK_SIZE>;

fn create_test_chunk(
    fill_type: TestBlock,
) -> Box<[[[TestBlock; TEST_CHUNK_SIZE]; TEST_CHUNK_SIZE]; TEST_CHUNK_SIZE]> {
    Box::new([[[fill_type; TEST_CHUNK_SIZE]; TEST_CHUNK_SIZE]; TEST_CHUNK_SIZE])
}

fn assert_aabbs_eq_unordered(mut actual: Vec<AABB>, mut expected: Vec<AABB>) {
    let sort_fn = |a: &AABB, b: &AABB| -> std::cmp::Ordering {
        let a_min = a.min.to_array();
        let b_min = b.min.to_array();
        a_min
            .partial_cmp(&b_min)
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    actual.sort_by(sort_fn);
    expected.sort_by(sort_fn);
    assert_eq!(actual, expected, "AABB lists do not match (ignoring order)");
}

#[test]
fn test_query_all_air_chunk() {
    let mut world = TestWorld::new();
    world.load_chunk(IVec3::ZERO, create_test_chunk(TestBlock::Air));
    let query_aabb = AABB::from_center_dims(Vec3::new(8.0, 8.0, 8.0), Vec3::ONE * 10.0);
    let result = world.query_potential_colliders(query_aabb);
    assert!(
        result.is_empty(),
        "Query in all-air chunk should return no colliders"
    );
}

#[test]
fn test_query_single_solid_block_precise() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[1][2][3] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 3.0, 4.0));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![AABB::new(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(2.0, 3.0, 4.0),
    )];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_single_solid_block_partial_overlap() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[1][2][3] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(1.5, 2.5, 3.5), Vec3::new(2.5, 3.5, 4.5));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![AABB::new(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(2.0, 3.0, 4.0),
    )];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_single_solid_block_contained_query() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[5][5][5] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(5.1, 5.1, 5.1), Vec3::new(5.9, 5.9, 5.9));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![AABB::new(
        Vec3::new(5.0, 5.0, 5.0),
        Vec3::new(6.0, 6.0, 6.0),
    )];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_miss_solid_block() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[1][2][3] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(2.1, 2.0, 3.0), Vec3::new(3.1, 3.0, 4.0));
    let result = world.query_potential_colliders(query_aabb);

    assert!(
        result.is_empty(),
        "Query missing the block should return empty"
    );
}

#[test]
fn test_query_multiple_solid_blocks() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[4][4][4] = TestBlock::Solid;
    chunk_data[4][4][5] = TestBlock::Solid;
    chunk_data[5][4][4] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(4.1, 4.1, 4.1), Vec3::new(4.9, 4.9, 5.9));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![
        AABB::new(Vec3::new(4.0, 4.0, 4.0), Vec3::new(5.0, 5.0, 5.0)),
        AABB::new(Vec3::new(4.0, 4.0, 5.0), Vec3::new(5.0, 5.0, 6.0)),
    ];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_block_at_chunk_origin() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[0][0][0] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![AABB::new(Vec3::ZERO, Vec3::ONE)];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_block_at_chunk_max_corner() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    let max_idx = TEST_CHUNK_SIZE - 1;

    chunk_data[max_idx][max_idx][max_idx] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let world_pos = max_idx as f32;

    let query_aabb = AABB::new(Vec3::splat(world_pos + 0.5), Vec3::splat(world_pos + 1.5));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![AABB::new(
        Vec3::splat(world_pos),
        Vec3::splat(world_pos + 1.0),
    )];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_across_chunk_boundary_positive() {
    let mut world = TestWorld::new();
    let max_idx = TEST_CHUNK_SIZE - 1;

    let mut chunk_data0 = create_test_chunk(TestBlock::Air);
    chunk_data0[max_idx][0][0] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data0);

    let mut chunk_data1 = create_test_chunk(TestBlock::Air);
    chunk_data1[0][0][0] = TestBlock::Solid;
    world.load_chunk(IVec3::new(1, 0, 0), chunk_data1);

    let query_aabb = AABB::new(Vec3::new(15.5, -0.5, -0.5), Vec3::new(16.5, 0.5, 0.5));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![
        AABB::new(Vec3::new(15.0, 0.0, 0.0), Vec3::new(16.0, 1.0, 1.0)),
        AABB::new(Vec3::new(16.0, 0.0, 0.0), Vec3::new(17.0, 1.0, 1.0)),
    ];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_across_chunk_boundary_negative() {
    let mut world = TestWorld::new();
    let max_idx = TEST_CHUNK_SIZE - 1;

    let mut chunk_data_neg = create_test_chunk(TestBlock::Air);

    chunk_data_neg[max_idx][0][0] = TestBlock::Solid;
    world.load_chunk(IVec3::new(-1, 0, 0), chunk_data_neg);

    let mut chunk_data_zero = create_test_chunk(TestBlock::Air);
    chunk_data_zero[0][0][0] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data_zero);

    let query_aabb = AABB::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    let result = world.query_potential_colliders(query_aabb);
    let expected = vec![
        AABB::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 1.0)),
        AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
    ];

    assert_aabbs_eq_unordered(result, expected);
}

#[test]
fn test_query_on_block_edge() {
    let mut world = TestWorld::new();
    let mut chunk_data = create_test_chunk(TestBlock::Air);
    chunk_data[1][1][1] = TestBlock::Solid;
    chunk_data[2][1][1] = TestBlock::Solid;
    world.load_chunk(IVec3::ZERO, chunk_data);

    let query_aabb = AABB::new(Vec3::new(2.0, 1.0, 1.0), Vec3::new(2.5, 1.5, 1.5));
    let result = world.query_potential_colliders(query_aabb);

    let expected = vec![AABB::new(
        Vec3::new(2.0, 1.0, 1.0),
        Vec3::new(3.0, 2.0, 2.0),
    )];
    assert_aabbs_eq_unordered(result, expected);

    let query_aabb_max = AABB::new(Vec3::new(1.5, 1.0, 1.0), Vec3::new(2.0, 1.5, 1.5));
    let result_max = world.query_potential_colliders(query_aabb_max);

    let expected_max = vec![AABB::new(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(2.0, 2.0, 2.0),
    )];
    assert_aabbs_eq_unordered(result_max, expected_max);
}
