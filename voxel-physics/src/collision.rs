use crate::aabb::AABB;
use glam::Vec3;

/// Swept `PhysicsBody` `AABB` vs Static `AABB` collision test.
/// Returns: `Option<(collision_time, obstacle_normal)>`
/// `obstacle_normal` is Vec3::ZERO if AABBs already intersect.
pub fn swept_aabb_vs_aabb(
    body_aabb: &AABB,
    velocity: Vec3,
    obstacle_aabb: &AABB,
) -> Option<(f32, Vec3)> {
    if body_aabb.intersects(obstacle_aabb) {
        return None;
    }

    let mut inv_entry = Vec3::ZERO;
    let mut inv_exit = Vec3::ZERO;

    for i in 0..3 {
        if velocity[i].abs() < f32::EPSILON {
            if body_aabb.max[i] <= obstacle_aabb.min[i] || body_aabb.min[i] >= obstacle_aabb.max[i]
            {
                return None;
            } else {
                inv_entry[i] = f32::NEG_INFINITY;
                inv_exit[i] = f32::INFINITY;
            }
        } else {
            let entry = (obstacle_aabb.min[i] - body_aabb.max[i]) / velocity[i];
            let exit = (obstacle_aabb.max[i] - body_aabb.min[i]) / velocity[i];
            inv_entry[i] = entry.min(exit);
            inv_exit[i] = entry.max(exit);
        }
    }

    let latest_entry = inv_entry.x.max(inv_entry.y).max(inv_entry.z);
    let earliest_exit = inv_exit.x.min(inv_exit.y).min(inv_exit.z);

    if latest_entry > earliest_exit || latest_entry >= 1.0 || latest_entry < 0.0 {
        return None;
    }

    let mut normal = Vec3::ZERO;
    if inv_entry.x >= latest_entry - f32::EPSILON {
        // Use tolerance comparison
        normal.x = if velocity.x <= 0.0 { 1.0 } else { -1.0 };
    } else if inv_entry.y >= latest_entry - f32::EPSILON {
        normal.y = if velocity.y <= 0.0 { 1.0 } else { -1.0 };
    } else {
        normal.z = if velocity.z <= 0.0 { 1.0 } else { -1.0 };
    }

    Some((latest_entry, normal))
}
