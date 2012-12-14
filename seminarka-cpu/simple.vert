#version 330 core

uniform mat4 u_ModelViewMatrix;
uniform mat4 u_ProjectionMatrix;

in vec4 a_Position;
//in vec4 a_Normal;

out block {
	vec4 v_Color;
	vec3 v_viewPos;
	vec4 v_Position;
} Out;

const vec4 color = vec4(0.4, 0.6, 1.0, 0.30);

void main () {

	vec4 viewPos = u_ModelViewMatrix * a_Position;//gl_Vertex;
	//v_Normal = normalize(mat3(u_ModelViewMatrix) * gl_Normal);
	Out.v_viewPos = -viewPos.xyz;
	
	Out.v_Position = u_ProjectionMatrix * viewPos;
	Out.v_Color = color;

}
