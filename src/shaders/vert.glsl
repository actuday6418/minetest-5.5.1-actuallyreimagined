#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_position;
layout(location = 2) out vec2 v_tex_coord;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec3 sun;
} uniforms;

void main() {
    mat4 view_proj = uniforms.proj * uniforms.view;
    gl_Position = view_proj * vec4(position, 1.0);

    v_normal = normal;

    v_position = position;
    v_tex_coord = tex_coord;
}
