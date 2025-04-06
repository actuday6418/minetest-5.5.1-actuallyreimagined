#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;
layout(location = 2) in vec3 v_sun;
layout(location = 0) out vec4 f_color;

const vec3 light_color = vec3(1.0, 1.0, 1.0);
const vec3 object_color = vec3(0.6, 0.7, 1.0);
const float ambient_strength = 0.15;

void main() {
    vec3 norm = normalize(v_normal);
    float dot_product = dot(norm, v_sun);
    float diffuse_intensity_scaled = (dot_product + 1.0) * 0.5;
    float total_intensity = clamp(diffuse_intensity_scaled + ambient_strength, 0.0, 1.0);
    vec3 final_color = light_color * total_intensity * object_color;
    f_color = vec4(final_color, 1.0);
}
