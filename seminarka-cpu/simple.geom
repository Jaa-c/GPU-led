#version 330 compatibility

layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;
//layout (line_strip, max_vertices=2) out;

in block{
	vec4 v_Color;
	vec4 v_Position;
	//vec3 v_viewPos;
} In[];


out block{
	vec4 v_Color;
	vec3 v_Position;
	vec3 v_Normal;
} Out;


const float SCALE = 15.0f;

/**
 * Only computing normals
 */
void main() {
	
	vec3 n = normalize(-cross((In[2].v_Position - In[0].v_Position).xyz, (In[1].v_Position - In[0].v_Position).xyz));

	for (int i = 0; i < gl_in.length(); i++) {
		Out.v_Color = In[i].v_Color;
		Out.v_Normal = n; 
		//Out.v_viewPos = In[i].v_viewPos;
		Out.v_Position = In[i].v_Position.xyz;
		gl_Position = In[i].v_Position;
		EmitVertex();
	}
	EndPrimitive();

	/* Vykreslení normál: * /
	for(int i = 0; i < gl_in.length(); i++){
		gl_Position = In[i].v_Position;
		Out.v_Normal = n;
		Out.v_viewPos = In[i].v_viewPos;
		Out.v_Color =  In[i].v_Color;
		EmitVertex();

		gl_Position = In[i].v_Position +  vec4(n , 0.0) * SCALE;
		Out.v_Normal = n;
		Out.v_viewPos = In[i].v_viewPos;
		Out.v_Color = vec4(1.0, 0.0, 0.0, 0.0);//In[i].v_Color;
		EmitVertex();

		EndPrimitive();
	}
	/**/

}