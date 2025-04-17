struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
};

struct QuadTemplate {
    model_positions: array<vec4<f32>, 4>, 
    uvs: array<vec2<f32>, 4>,
    normal: vec4<f32>, 
};

struct PushConstants {
    chunk_offset_world: vec3<f32>,
    
};

struct QuadBuffer {
    data: array<QuadTemplate>, 
};

struct VertexInput {
    @location(0) block_position: vec4<u32>, 
    @location(1) quad_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,       
    @location(1) world_pos: vec3<f32>,    
    @location(2) tex_coord: vec2<f32>,    
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;
@group(1) @binding(2) var<storage, read> quad_buffer: QuadBuffer; 

var<push_constant> push_constants: PushConstants;

@vertex
fn vs_main(
    instance: VertexInput, 
    @builtin(vertex_index) vertex_index: u32 
) -> VertexOutput {

    let corner_indices = array<u32, 6>(0u, 1u, 2u, 2u, 3u, 0u);
    let quad_corner_index = corner_indices[vertex_index];

    let quad: QuadTemplate = quad_buffer.data[instance.quad_index];

    let model_pos: vec3<f32> = quad.model_positions[quad_corner_index].xyz; 
    let uv: vec2<f32> = quad.uvs[quad_corner_index]; 
    let normal: vec3<f32> = quad.normal.xyz;
  
    let block_offset_local = vec3<f32>(f32(instance.block_position.x), f32(instance.block_position.y), f32(instance.block_position.z));
    
    let world_pos = push_constants.chunk_offset_world + block_offset_local + vec3(0.5) + model_pos;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.normal = normal; 
    out.world_pos = world_pos;
    out.tex_coord = uv;

    return out;
}

const LIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const AMBIENT_STRENGTH: f32 = 0.15;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coord);
       
    let norm = normalize(in.normal);
    let sun_dir = normalize(uniforms.sun_direction);
    
    let dot_product = dot(norm, sun_dir); 
    let diffuse_intensity_scaled = (dot_product + 1.0) * 0.5; 
    
    let total_intensity = clamp(diffuse_intensity_scaled + AMBIENT_STRENGTH, 0.0, 1.0);
    let final_color_rgb = LIGHT_COLOR * total_intensity * tex_color.rgb;

    return vec4<f32>(final_color_rgb, tex_color.a);
}
