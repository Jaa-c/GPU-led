#version 330 core

//uniform mat4 u_ModelViewMatrix;

in block {
	vec4 v_Color;
	vec3 v_Position;
	vec3 v_Normal;
} In;

// light
const vec3 light_pos = vec3(0.0, 0.0, 1.0);
const float diffIntensity = 1.4f;
const float specIntensity = 0.2f;

/**
 * Only phong lightning
 */
void main() {

	vec3 N = In.v_Normal;
	vec3 E = normalize(In.v_Position);
	vec3 L = normalize(light_pos - In.v_Position);
	vec3 R = normalize(reflect(L, -N));
	
	float diffuse = max(dot(N, L), 0.0);
	float specular = pow(max(dot(R, E), 0.0), 64.0);

	vec3 color = In.v_Color.xyz * diffuse * diffIntensity + specular * specIntensity;
	
	gl_FragColor = vec4(color, 1.0);

}
