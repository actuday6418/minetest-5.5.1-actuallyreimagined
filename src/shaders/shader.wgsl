struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
};

struct QuadTemplate {
    model_positions: array<vec4<f32>, 4>,
    uvs: array<vec2<f32>, 4>,
    normal_index: u32,
};

struct PushConstants {
    chunk_offset_world: vec3<f32>,

};

struct QuadBuffer {
    data: array<QuadTemplate>,
};

struct VertexInput {
    @location(0) packed_origin: u32,
    @location(1) packed_scale_quad_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(3) quad_idx: u32, 
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

    let origin_packed = instance.packed_origin;
    let block_offset_local_x = f32(origin_packed & 0xFFu);
    let block_offset_local_y = f32((origin_packed >> 8u) & 0xFFu);
    let block_offset_local_z = f32((origin_packed >> 16u) & 0xFFu);
    let block_offset_local = vec3<f32>(block_offset_local_x, block_offset_local_y, block_offset_local_z);
   
    let scale_etc_packed = instance.packed_scale_quad_index;
    let u_scale_minus_1 = scale_etc_packed & 0x3Fu;       
    let v_scale_minus_1 = (scale_etc_packed >> 6u) & 0x3Fu; 
    let quad_index = scale_etc_packed >> 12u;            
   
    let quad: QuadTemplate = quad_buffer.data[quad_index];
   
    let u_size = f32(u_scale_minus_1 + 1u); 
    let v_size = f32(v_scale_minus_1 + 1u); 

    let normal_idx = quad.normal_index; 

    let corner_indices_std = array<u32, 6>(0u, 1u, 2u, 2u, 3u, 0u);
    let corner_indices_flipped = array<u32, 6>(0u, 2u, 1u, 2u, 0u, 3u);

    var corner_indices: array<u32, 6>;
    if (normal_idx == 0 || normal_idx == 2 || normal_idx == 5) {
        corner_indices = corner_indices_flipped;
    } else {
        corner_indices = corner_indices_std;
    }
    let quad_corner_index = corner_indices[vertex_index];

    var normal: vec3<f32>;
    
    var corner_offset_u: f32; 
    var corner_offset_v: f32; 
    
    if (quad_corner_index == 0u) { 
        corner_offset_u = 0.0; corner_offset_v = 0.0;
    } else if (quad_corner_index == 1u) { 
        corner_offset_u = u_size; corner_offset_v = 0.0;
    } else if (quad_corner_index == 2u) { 
        corner_offset_u = u_size; corner_offset_v = v_size;
    } else { 
        corner_offset_u = 0.0; corner_offset_v = v_size;
    }

   
    var final_local_pos: vec3<f32>;
    if (normal_idx == 0u || normal_idx == 1u) { 
        normal = vec3(0.0, 0.0, select(1.0, -1.0, normal_idx == 1u));
        final_local_pos = block_offset_local + vec3<f32>(corner_offset_u, corner_offset_v, 0.0);
    } else if (normal_idx == 2u || normal_idx == 3u) { 
        normal = vec3<f32>(select(1.0, -1.0, normal_idx == 2u), 0.0, 0.0);
        final_local_pos = block_offset_local + vec3<f32>(0.0, corner_offset_v, corner_offset_u);
    } else { 
        normal = vec3<f32>(0.0, select(-1.0, 1.0, normal_idx == 5u), 0.0);
        final_local_pos = block_offset_local + vec3<f32>(corner_offset_u, 0.0, corner_offset_v);
    }

    let world_pos = push_constants.chunk_offset_world + final_local_pos;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.normal = normal;
    out.world_pos = world_pos;
    out.quad_idx = quad_index; 

    return out;
}

const LIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const AMBIENT_STRENGTH: f32 = 0.15;
const ATLAS_W: f32 = 1024.0;
const ATLAS_H: f32 = 1024.0;

struct FragmentInput {
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(3) quad_idx: u32, 
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
   
    var tile_uv: vec2<f32>;
    let norm_abs = abs(in.normal);

    if (norm_abs.y > norm_abs.x && norm_abs.y > norm_abs.z) { 
        tile_uv = fract(in.world_pos.xz); 
    } else if (norm_abs.z > norm_abs.x) { 
        tile_uv = fract(in.world_pos.xy); 
    } else { 
        tile_uv = vec2<f32>(fract(in.world_pos.z), fract(in.world_pos.y)); 
    }

    if (!(norm_abs.y > norm_abs.x && norm_abs.y > norm_abs.z)) {
         tile_uv.y = 1.0 - tile_uv.y;
    }
    
    let quad_template: QuadTemplate = quad_buffer.data[in.quad_idx];

    let uv0 = quad_template.uvs[0];
    let uv1 = quad_template.uvs[1];
    let uv2 = quad_template.uvs[2];
    let uv3 = quad_template.uvs[3];
    let min_atlas_uv_orig = vec2(min(min(uv0.x, uv1.x), min(uv2.x, uv3.x)), min(min(uv0.y, uv1.y), min(uv2.y, uv3.y)));
    let max_atlas_uv_orig = vec2(max(max(uv0.x, uv1.x), max(uv2.x, uv3.x)), max(max(uv0.y, uv1.y), max(uv2.y, uv3.y)));
    let dimensions_uv_orig = max_atlas_uv_orig - min_atlas_uv_orig;
    
    let min_atlas_uv = vec2(min(min(uv0.x, uv1.x), min(uv2.x, uv3.x)),
                           min(min(uv0.y, uv1.y), min(uv2.y, uv3.y)));
    let max_atlas_uv = vec2(max(max(uv0.x, uv1.x), max(uv2.x, uv3.x)),
                           max(max(uv0.y, uv1.y), max(uv2.y, uv3.y)));

    let dimensions_uv = max_atlas_uv - min_atlas_uv; 
    let final_atlas_uv: vec2<f32> = min_atlas_uv + tile_uv * dimensions_uv;

    let tex_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, final_atlas_uv);

    let norm = normalize(in.normal);
    let sun_dir = normalize(uniforms.sun_direction);
    let dot_product = dot(norm, sun_dir);
    let diffuse_intensity_scaled = (dot_product + 1.0) * 0.5;
    let total_intensity = clamp(diffuse_intensity_scaled + AMBIENT_STRENGTH, 0.0, 1.0);
    let final_color_rgb = LIGHT_COLOR * total_intensity * tex_color.rgb;

    return vec4<f32>(final_color_rgb, tex_color.a);
}
