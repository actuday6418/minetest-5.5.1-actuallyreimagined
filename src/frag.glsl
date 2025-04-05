#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;
layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

void main() {
    vec3 temp = v_position * 0.75;
    f_color = vec4(temp[0], temp[1], temp[2], 1.0);
}
