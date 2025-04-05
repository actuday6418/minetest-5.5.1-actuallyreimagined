#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_normal; // Pass world-space normal

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camera_pos; // World space
    vec3 light_pos; // World space
} ubo;

void main() {
    mat4 model_view_proj = ubo.proj * ubo.view * ubo.model;
    gl_Position = model_view_proj * vec4(position, 1.0);

    // Calculate world position
    out_world_pos = vec3(ubo.model * vec4(position, 1.0));

    // Calculate world-space normal (correctly handling non-uniform scaling)
    // Use the inverse transpose of the model matrix's upper-left 3x3
    out_normal = normalize(mat3(transpose(inverse(ubo.model))) * normal);
}
