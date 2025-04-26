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
        return Some((0.0, Vec3::ZERO));
    }

    let inv_entry = Vec3::new(
        if velocity.x == 0.0 {
            f32::NEG_INFINITY
        } else {
            (obstacle_aabb.min.x - body_aabb.max.x) / velocity.x
        },
        if velocity.y == 0.0 {
            f32::NEG_INFINITY
        } else {
            (obstacle_aabb.min.y - body_aabb.max.y) / velocity.y
        },
        if velocity.z == 0.0 {
            f32::NEG_INFINITY
        } else {
            (obstacle_aabb.min.z - body_aabb.max.z) / velocity.z
        },
    );
    let inv_exit = Vec3::new(
        if velocity.x == 0.0 {
            f32::INFINITY
        } else {
            (obstacle_aabb.max.x - body_aabb.min.x) / velocity.x
        },
        if velocity.y == 0.0 {
            f32::INFINITY
        } else {
            (obstacle_aabb.max.y - body_aabb.min.y) / velocity.y
        },
        if velocity.z == 0.0 {
            f32::INFINITY
        } else {
            (obstacle_aabb.max.z - body_aabb.min.z) / velocity.z
        },
    );

    let entry_time = Vec3::new(
        inv_entry.x.min(inv_exit.x),
        inv_entry.y.min(inv_exit.y),
        inv_entry.z.min(inv_exit.z),
    );
    let exit_time = Vec3::new(
        inv_entry.x.max(inv_exit.x),
        inv_entry.y.max(inv_exit.y),
        inv_entry.z.max(inv_exit.z),
    );

    let latest_entry = entry_time.x.max(entry_time.y).max(entry_time.z);
    let earliest_exit = exit_time.x.min(exit_time.y).min(exit_time.z);

    if latest_entry > earliest_exit || latest_entry >= 1.0 || latest_entry < 0.0 {
        return None;
    }

    let mut normal = Vec3::ZERO;
    if entry_time.x == latest_entry {
        normal.x = if velocity.x < 0.0 { 1.0 } else { -1.0 };
    } else if entry_time.y == latest_entry {
        normal.y = if velocity.y < 0.0 { 1.0 } else { -1.0 };
    } else {
        normal.z = if velocity.z < 0.0 { 1.0 } else { -1.0 };
    }

    Some((latest_entry, normal))
}
