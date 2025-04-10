#version 450

layout(location = 0) in uvec3 block_position;
layout(location = 1) in uint quad_index;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world_pos;
layout(location = 2) out vec2 v_tex_coord;

struct QuadTemplate {
    vec4 model_positions[4];
    vec2 uvs[4];
    vec4 normal;
};

layout(set = 0, binding = 0) uniform Data {
    mat4 view;
    mat4 proj;
    vec3 sun;
} uniforms;

layout(push_constant) uniform PushConstants {
    vec3 chunk_offset_world;
} push_constants;

layout(set = 0, binding = 1) uniform sampler2D tex_sampler;

layout(set = 0, binding = 2, std430) readonly buffer QuadBuffer {
    QuadTemplate data[];
} quadBuffer;

void main() {
    QuadTemplate quad = quadBuffer.data[quad_index];

    vec3 model_pos = quad.model_positions[gl_VertexIndex].xyz;
    vec2 uv = quad.uvs[gl_VertexIndex];
    vec3 normal = quad.normal.xyz;

    vec3 block_offset_local = vec3(block_position);
    vec3 world_pos = push_constants.chunk_offset_world + block_offset_local + vec3(0.5) + model_pos;
    mat4 view_proj = uniforms.proj * uniforms.view;
    gl_Position = view_proj * vec4(world_pos, 1.0);

    v_normal = normal;
    v_world_pos = world_pos;
    v_tex_coord = uv;
}
