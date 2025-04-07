#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;
layout(location = 2) in vec3 v_sun;
layout(location = 3) in vec2 v_tex_coord;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex_sampler;

const vec3 light_color = vec3(1.0, 1.0, 1.0);
const float ambient_strength = 0.15;

void main() {
    vec4 tex_color = texture(tex_sampler, v_tex_coord);

    vec3 norm = normalize(v_normal);
    float dot_product = dot(norm, v_sun);
    float diffuse_intensity_scaled = (dot_product + 1.0) * 0.5;
    float total_intensity = clamp(diffuse_intensity_scaled + ambient_strength, 0.0, 1.0);

    vec3 final_color = light_color * total_intensity * tex_color.rgb;
    f_color = vec4(final_color, tex_color.a);
}
