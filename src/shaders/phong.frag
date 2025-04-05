#version 450

layout(location = 0) in vec3 world_pos;    // Fragment position in world space
layout(location = 1) in vec3 world_normal; // Interpolated normal in world space

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camera_pos; // World space
    vec3 light_pos; // World space - Make sure this matches the struct!
} ubo;

void main() {
    // Material properties (hardcoded for simplicity)
    vec3 object_color = vec3(0.2, 0.6, 1.0); // A nice blue
    float ambient_strength = 0.1;
    float diffuse_strength = 0.8;
    float specular_strength = 0.9; // Strong highlights
    float shininess = 64.0; // Controls highlight size

    // Light properties (hardcoded)
    vec3 light_color = vec3(1.0, 1.0, 1.0); // White light

    // Normalize vectors
    vec3 norm = normalize(world_normal);
    // Use ubo.light_pos from the uniform buffer
    vec3 light_dir = normalize(ubo.light_pos - world_pos);
    vec3 view_dir = normalize(ubo.camera_pos - world_pos);

    // Ambient component
    vec3 ambient = ambient_strength * light_color;

    // Diffuse component
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diffuse_strength * diff * light_color;

    // Specular component (Phong)
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular = specular_strength * spec * light_color;

    // Combine components
    vec3 result = (ambient + diffuse + specular) * object_color;

    // Output final color
    f_color = vec4(result, 1.0);
}
