#version 330 core

uniform mat4 u_ModelViewMatrix;

in block {
	vec4 v_Color;
	vec3 v_viewPos;
	vec3 v_Normal;
} In;

const vec3 light_pos = vec3(0.0, 0.0, 1.0);

void main() {

	//gl_FragColor = In.v_Color;
	//return;
	
	vec3 N = In.v_Normal;
	vec3 E = normalize(In.v_viewPos);
	vec3 L = normalize(light_pos - In.v_viewPos);
	vec3 R = normalize(reflect(L, -N));
	
	float diffuse = max(dot(N, L), 0.0);
	float specular = 0.0f;//pow(max(dot(R, E), 0.0), 128.0);

	vec3 color = In.v_Color.xyz * diffuse + specular;
	
	gl_FragColor = vec4(color, 1.0);//v_Color.w);

}
