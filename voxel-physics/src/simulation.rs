use crate::aabb::AABB;
use crate::body::PhysicsBody;
use crate::collision::swept_aabb_vs_aabb;
use crate::world::PhysicsWorldProvider;
use glam::Vec3;

/// Applies velocity, detects collisions using swept AABB tests against potential
/// colliders provided by the `PhysicsWorldProvider`.
pub fn step_simulation<W: PhysicsWorldProvider>(body: &mut PhysicsBody, dt: f32, world: &W) {
    if dt <= 0.0 {
        return;
    }

    let start_position = body.position;
    let start_aabb = body.get_world_aabb();
    let velocity_this_frame = body.velocity * dt;
    let target_position = start_position + velocity_this_frame;

    let target_aabb = body.get_world_aabb_at(target_position);
    let swept_aabb = start_aabb.union(&target_aabb);

    let potential_colliders = world.query_potential_colliders(swept_aabb);

    let mut min_collision_time = 1.0_f32;
    let mut final_collision_normal = Vec3::ZERO;
    let mut overlapping_obstacle: Option<AABB> = None;

    for obstacle_aabb in &potential_colliders {
        if let Some((time, normal)) =
            swept_aabb_vs_aabb(&start_aabb, velocity_this_frame, obstacle_aabb)
        {
            if time == 0.0 && min_collision_time > 0.0 {
                min_collision_time = 0.0;
                overlapping_obstacle = Some(*obstacle_aabb);
                final_collision_normal = Vec3::ZERO;
                break;
            } else if time > 0.0 && time < min_collision_time {
                min_collision_time = time;
                final_collision_normal = normal;
                overlapping_obstacle = None;
            }
        }
    }

    let epsilon_push = 0.0001 * dt;
    let actual_move_time = (min_collision_time - epsilon_push).max(0.0);
    let actual_move_vector = velocity_this_frame * actual_move_time;
    body.position += actual_move_vector;

    body.is_grounded = false;

    if min_collision_time < 1.0 {
        let mut response_normal = final_collision_normal;

        if min_collision_time == 0.0 {
            if let Some(obstacle) = overlapping_obstacle {
                let body_center = start_aabb.center();
                let obstacle_center = obstacle.center();
                let center_diff = body_center - obstacle_center;

                let abs_diff = center_diff.abs();
                if abs_diff.x >= abs_diff.y && abs_diff.x >= abs_diff.z {
                    response_normal = Vec3::X * center_diff.x.signum();
                } else if abs_diff.y >= abs_diff.x && abs_diff.y >= abs_diff.z {
                    response_normal = Vec3::Y * center_diff.y.signum();
                } else {
                    response_normal = Vec3::Z * center_diff.z.signum();
                }
            }
        }

        let tolerance = 0.01;

        if response_normal.length_squared() > 0.5 {
            if (response_normal.x.abs() - 1.0).abs() < tolerance {
                body.velocity.x = 0.0;
            }
            if (response_normal.y.abs() - 1.0).abs() < tolerance {
                if response_normal.y > 0.0 {
                    body.is_grounded = true;
                }
                body.velocity.y = 0.0;
            }
            if (response_normal.z.abs() - 1.0).abs() < tolerance {
                body.velocity.z = 0.0;
            }
        }
    }
}
