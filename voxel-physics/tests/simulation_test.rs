use glam::Vec3;
use voxel_physics::{AABB, PhysicsBody, PhysicsWorldProvider, step_simulation};

const DELTA_TIME: f32 = 1.0 / 60.0;
const PLAYER_DIMS: Vec3 = Vec3::new(1.0, 2.0, 1.0);
const PLAYER_HALF_HEIGHT: f32 = PLAYER_DIMS.y * 0.5;
const PLAYER_HALF_WIDTH: f32 = PLAYER_DIMS.x * 0.5;
const PLAYER_HALF_DEPTH: f32 = PLAYER_DIMS.z * 0.5;

fn assert_vec3_approx_eq(a: Vec3, b: Vec3, tolerance: f32) {
    assert!(
        (a - b).length_squared() < tolerance * tolerance,
        "Assertion failed: {:?} != {:?} within tolerance {}",
        a,
        b,
        tolerance
    );
}

struct MockWorld {
    colliders: Vec<AABB>,
}

impl MockWorld {
    fn new(colliders: Vec<AABB>) -> Self {
        Self { colliders }
    }
}

impl PhysicsWorldProvider for MockWorld {
    fn query_potential_colliders(&self, query_aabb: AABB) -> Vec<AABB> {
        self.colliders
            .iter()
            .filter(|&collider_aabb| query_aabb.intersects(collider_aabb))
            .cloned()
            .collect()
    }
}

#[test]
fn test_fall_onto_block() {
    let block = AABB::new(Vec3::ZERO, Vec3::ONE);
    let world = MockWorld::new(vec![block]);

    let mut player = PhysicsBody::new(Vec3::new(0.5, 3.0, 0.5), PLAYER_DIMS);
    player.velocity = Vec3::new(0.0, -100.0, 0.0);

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos_y = 1.0 + PLAYER_HALF_HEIGHT;
    let expected_pos = Vec3::new(0.5, expected_pos_y, 0.5);

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);
    assert_eq!(player.velocity.y, 0.0, "Vertical velocity should be zeroed");
    assert!(player.is_grounded, "Player should be grounded");
    assert_eq!(player.velocity.x, 0.0);
    assert_eq!(player.velocity.z, 0.0);
}

#[test]
fn test_move_into_wall_head_on() {
    let wall = AABB::new(Vec3::new(3.0, -10.0, -10.0), Vec3::new(4.0, 10.0, 10.0));
    let world = MockWorld::new(vec![wall]);

    let mut player = PhysicsBody::new(Vec3::new(1.0, 1.0, 0.0), PLAYER_DIMS);
    player.velocity = Vec3::new(100.0, 0.0, 0.0);

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos_x = 3.0 - PLAYER_HALF_WIDTH;
    let expected_pos = Vec3::new(expected_pos_x, 1.0, 0.0);

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);
    assert_eq!(
        player.velocity.x, 0.0,
        "Horizontal velocity X should be zeroed"
    );
    assert_eq!(player.velocity.y, 0.0);
    assert_eq!(player.velocity.z, 0.0);
    assert!(!player.is_grounded);
}

#[test]
fn test_move_into_wall_diagonally() {
    let wall = AABB::new(Vec3::new(3.0, -10.0, -10.0), Vec3::new(4.0, 10.0, 10.0));
    let world = MockWorld::new(vec![wall]);

    let initial_vel = Vec3::new(100.0, 0.0, 5.0);
    let mut player = PhysicsBody::new(Vec3::new(1.0, 1.0, 0.0), PLAYER_DIMS);
    player.velocity = initial_vel;

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos_x = 3.0 - PLAYER_HALF_WIDTH;

    let time_to_hit = (expected_pos_x - 1.0) / initial_vel.x;
    let expected_pos_z = 0.0 + initial_vel.z * time_to_hit;

    let expected_pos = Vec3::new(expected_pos_x, 1.0, expected_pos_z);

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);
    assert_eq!(
        player.velocity.x, 0.0,
        "Horizontal velocity X should be zeroed"
    );
    assert_eq!(player.velocity.y, 0.0);

    assert_eq!(
        player.velocity.z, initial_vel.z,
        "Z velocity should persist"
    );
    assert!(!player.is_grounded);
}

#[test]
fn test_move_into_corner_diagonally() {
    let wall_x = AABB::new(Vec3::new(3.0, -10.0, -10.0), Vec3::new(4.0, 10.0, 10.0));
    let wall_z = AABB::new(Vec3::new(-10.0, -10.0, 3.0), Vec3::new(10.0, 10.0, 4.0));
    let world = MockWorld::new(vec![wall_x, wall_z]);

    let initial_vel = Vec3::new(100.0, 0.0, 10.0);
    let mut player = PhysicsBody::new(Vec3::new(1.0, 1.0, 1.0), PLAYER_DIMS);
    player.velocity = initial_vel;

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos_x = 3.0 - PLAYER_HALF_WIDTH;
    let time_to_hit_x = (expected_pos_x - 1.0) / initial_vel.x;

    let expected_pos_z = 1.0 + initial_vel.z * time_to_hit_x;
    let expected_pos = Vec3::new(expected_pos_x, 1.0, expected_pos_z);

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);

    assert_eq!(player.velocity.x, 0.0, "X velocity should be zeroed");
    assert_eq!(player.velocity.y, 0.0);
    assert_eq!(
        player.velocity.z, initial_vel.z,
        "Z velocity should persist"
    );
    assert!(!player.is_grounded);
}

#[test]
fn test_fall_freely() {
    let world = MockWorld::new(vec![]);

    let start_pos = Vec3::new(0.0, 10.0, 0.0);
    let initial_vel = Vec3::new(0.0, -10.0, 0.0);
    let mut player = PhysicsBody::new(start_pos, PLAYER_DIMS);
    player.velocity = initial_vel;

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos = start_pos + initial_vel * DELTA_TIME;

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);

    assert_eq!(player.velocity, initial_vel);
    assert!(!player.is_grounded);
}

#[test]
fn test_move_freely_horizontally() {
    let world = MockWorld::new(vec![]);

    let start_pos = Vec3::new(0.0, 1.0, 0.0);
    let initial_vel = Vec3::new(5.0, 0.0, 2.0);
    let mut player = PhysicsBody::new(start_pos, PLAYER_DIMS);
    player.velocity = initial_vel;

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos = start_pos + initial_vel * DELTA_TIME;

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);
    assert_eq!(player.velocity, initial_vel);
    assert!(!player.is_grounded);
}

#[test]
fn test_start_overlapping_block() {
    let block = AABB::new(Vec3::ZERO, Vec3::ONE);
    let world = MockWorld::new(vec![block]);

    let start_pos_y = 1.0 - 0.1 + PLAYER_HALF_HEIGHT;
    let mut player = PhysicsBody::new(Vec3::new(0.5, start_pos_y, 0.5), PLAYER_DIMS);
    player.velocity = Vec3::new(0.0, -1.0, 0.0);

    step_simulation(&mut player, DELTA_TIME, &world);

    let expected_pos = Vec3::new(0.5, start_pos_y, 0.5);

    assert_vec3_approx_eq(player.position, expected_pos, 0.001);
    assert_eq!(
        player.velocity.y, 0.0,
        "Vertical velocity should be zeroed due to overlap"
    );

    assert!(player.is_grounded, "Player should likely be grounded");
}
