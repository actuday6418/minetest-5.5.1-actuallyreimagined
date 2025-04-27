use crate::body::PhysicsBody;
use crate::collision::swept_aabb_vs_aabb;
use crate::world::PhysicsWorldProvider;
use glam::Vec3;

fn move_along_axis<W: PhysicsWorldProvider>(
    body: &mut PhysicsBody,
    axis_index: usize,
    distance: f32,
    world: &W,
) -> f32 {
    if distance.abs() < f32::EPSILON {
        return 0.0;
    }

    let mut move_vec = Vec3::ZERO;
    move_vec[axis_index] = distance;

    let start_aabb = body.get_world_aabb();
    let target_aabb = start_aabb.translate(move_vec);
    let swept_aabb = start_aabb.union(&target_aabb);

    let potential_colliders = world.query_potential_colliders(swept_aabb);

    let mut min_collision_time = 1.0;
    let mut collision_normal_axis = 0.0;

    for obstacle_aabb in &potential_colliders {
        if let Some((time, normal)) = swept_aabb_vs_aabb(&start_aabb, move_vec, obstacle_aabb) {
            if time >= 0.0 && time < min_collision_time && normal[axis_index].abs() > 0.1 {
                min_collision_time = time;
                collision_normal_axis = normal[axis_index];
            }
        }
    }

    let epsilon_push = 1e-5;
    let actual_move_fraction = (min_collision_time - epsilon_push).max(0.0);
    let actual_move_dist = distance * actual_move_fraction;

    body.position[axis_index] += actual_move_dist;

    if min_collision_time < 1.0 {
        let moving_positive = distance > 0.0;
        let normal_opposes = (moving_positive && collision_normal_axis < -0.1)
            || (!moving_positive && collision_normal_axis > 0.1);

        if normal_opposes {
            body.velocity[axis_index] = 0.0;
            if axis_index == 1 && collision_normal_axis > 0.5 {
                body.is_grounded = true;
            }
        }
    }

    actual_move_dist
}

/// Applies velocity, detects collisions using swept AABB tests against potential
/// colliders provided by the `PhysicsWorldProvider`.
pub fn step_simulation<W: PhysicsWorldProvider>(body: &mut PhysicsBody, dt: f32, world: &W) {
    if dt <= 0.0 {
        return;
    }
    body.is_grounded = false;
    let displacement = body.velocity * dt;
    move_along_axis(body, 0, displacement.x, world);
    move_along_axis(body, 2, displacement.z, world);
    move_along_axis(body, 1, displacement.y, world);
}
