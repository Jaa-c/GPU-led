//----------------------------------------------------------------------------------------
/**
 * @file       main.cpp
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  39GPU - seminární práce - simulace tání ledu na cpu a na gpu
 *
*/
//----------------------------------------------------------------------------------------

#define USE_ANTTWEAKBAR

#include <iostream>
#include <fstream>
#include <Windows.h>
#include <time.h>
#include "../common/common.h"
#include "CPUSimulation.h"
#include "GPUSimulation.h"
#include "CPUMarchingCubes.h"

/************************************************
*				Shader files			    *
************************************************/
const char* VS_FILE_NAME		= "simple.vert";  // Vertex shader source file
const char* GS_FILE_NAME		= "simple.geom";  // Geometry shader source file
const char* FS_FILE_NAME		= "simple.frag";  // Fragment shader source file

/************************************************
*				Default settings			    *
************************************************/
GLint    g_WindowWidth			= WINDOW_WIDTH;    // Window width
GLint    g_WindowHeight			= WINDOW_HEIGHT;    // Window height

GLfloat  g_SceneRot[]			= { 0.0f, 0.0f, 0.0f, 1.0f }; // Scene orientation
bool     g_SceneRotEnabled		= false;  // Scene auto-rotation enabled/disabled
bool     g_WireMode				= false;  // Wire mode enabled/disabled

bool     g_UseShaders			= true;  // Programmable pipeline on/off
bool     g_UseVertexShader		= true;  // Use vertex shader
bool     g_UseGeometryShader	= true;  // Use geometry shader
bool     g_UseFragmentShader	= true;  // Use fragment shader

bool     g_useMarchingCubes		= true;  // use geometry
bool     g_drawPoints			= false;  // use geometry
bool     g_melt					= false;  // to melt or not to melt?

GLfloat  g_SceneTraZ			= std::max(std::max(DATA_WIDTH, DATA_HEIGHT), DATA_DEPTH) * 1.4f; // Scene translation along z-axis
int realDataCount				= DATA_SIZE;

/************************************************
*				Deprecated			    *
************************************************/
GLfloat* dataPoints				= NULL;   //seznam bodu pro vykresleni

// GLSL variables
GLuint g_ProgramId = 0; // Shader program id

// FORWARD DECLARATIONS________________________________________________________
#ifdef USE_ANTTWEAKBAR
    void TW_CALL cbSetShaderStatus(const void*, void*);
    void TW_CALL cbGetShaderStatus(void*, void*);
#endif
void TW_CALL cbCompileShaderProgram(void *clientData);

/************************************************
*				Simulation			    *
************************************************/
Simulation* simulation;
int cycles;
int particles;
long begin, end;
long time_current, time_global;

void initGUI();

static bool init = true;
static bool initMelt = true;




std::ofstream outputTest;
//-----------------------------------------------------------------------------
// Name: cbDisplay()
// Desc: 
//-----------------------------------------------------------------------------
void cbDisplay()
{
    static GLfloat scene_rot = 0.0f;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, g_WireMode ? GL_LINE : GL_FILL);
	glDisable(GL_CULL_FACE);
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Setup camera
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -g_SceneTraZ);
    pgr2AddQuaternionRotationToTransformation(g_SceneRot);
    glRotatef(scene_rot, 0.0f, 1.0f, 0.0f);
	
	if(init) {
		init = false;
		glEnableVertexAttribArray(1);
		srand ( time(NULL) );
		simulation->init();

		if(TEST_OUTPUT) {
			if(COMPUTE_ON_GPU)
				outputTest.open ("test-gpu.txt");
			else
				outputTest.open ("test-cpu.txt");
		}
	}

	if(g_melt) {
		if(initMelt) {
			initMelt = false;
			cycles = CONSOLE_OUTPUT_CYCLES;
			time_global = 0;
			time_current = 0;
			
		}

		begin = timeGetTime();
		particles = simulation->updateParticles();
		end = timeGetTime();
		time_current += end - begin;

		if(particles == 0) {
			g_melt = false;
			cycles = CONSOLE_OUTPUT_CYCLES;
			outputTest.close();
		}

		if(CONSOLE_OUTPUT) {
			cycles++;
			if(cycles >= CONSOLE_OUTPUT_CYCLES) {
				cycles = 0;
				time_global += time_current;
				std::cout << "ice particles: " << particles << " time: " << time_current << "ms, sum: " << time_global << "ms\n";
				time_current = 0;
			}
		}

		if(TEST_OUTPUT) {
			Voxel * d = simulation->getData();
			Voxel v = d[TEST_VOXEL];
			if(v.status == ICE)
				outputTest << v.temperature << "\n";
		}

	}
	
    // Turn on programmable pipeline
    if (g_UseShaders)
    {
        glUseProgram(g_ProgramId);    // Active shader program
		
		//nahrajeme modelview a projection matrix do shaderu
		GLfloat matrix[16] = {0};
		glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
		glUniformMatrix4fv(glGetUniformLocation(g_ProgramId, "u_ModelViewMatrix"), 1, GL_FALSE, matrix);

		glGetFloatv(GL_PROJECTION_MATRIX, matrix);
		glUniformMatrix4fv(glGetUniformLocation(g_ProgramId, "u_ProjectionMatrix"), 1, GL_FALSE, matrix);
	
		if(g_useMarchingCubes) {
			glBegin(GL_TRIANGLES);
			simulation->march();
			glEnd();

			/** /
			GLint hVertex = glGetAttribLocation(g_ProgramId, "a_Position");
			glVertexAttribPointer(hVertex, 3, GL_FLOAT, GL_FALSE, 0, &triangleNet);
			glEnableVertexAttribArray(hVertex);
	
			glDrawArrays(GL_TRIANGLES, 0, triangleNetIndex/3);
	
			glDisableVertexAttribArray(hVertex);
			/**/
		}

    }

    // Turn off programmable pipeline
    glUseProgram(NULL);

    if (g_SceneRotEnabled)
    {
        scene_rot+=0.25;
    }
}

//-----------------------------------------------------------------------------
// Name: cbInitGL()
// Desc: 
//-----------------------------------------------------------------------------
void cbInitGL()
{
	//init data
	if(COMPUTE_ON_GPU)
		simulation = new GPUSimulation();
	else
		simulation = new CPUSimulation();

    // Init app GUI
    initGUI();

    // Set OpenGL state variables
    glClearColor(0.15f, 0.15f, 0.15f, 0);
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

    cbCompileShaderProgram(NULL);
}


//-----------------------------------------------------------------------------
// Name: cbCompileShaderProgram()
// Desc: 
//-----------------------------------------------------------------------------
void TW_CALL cbCompileShaderProgram(void *clientData)
{
    // Delete shader program if exists
    if (g_ProgramId)
    {
        glDeleteProgram(g_ProgramId);
    }

    // Create shader program object
    g_ProgramId = glCreateProgram();

    if (g_UseVertexShader)
    {
        // Create shader objects for vertex shader
        GLuint id = pgr2CreateShaderFromFile(GL_VERTEX_SHADER, VS_FILE_NAME);
        glAttachShader(g_ProgramId, id);
        glDeleteShader(id);
    }
    if (g_UseGeometryShader)
    {
        // Create shader objects for geometry shader
        GLuint id = pgr2CreateShaderFromFile(GL_GEOMETRY_SHADER, GS_FILE_NAME);
        glAttachShader(g_ProgramId, id);
        glDeleteShader(id);
        //glProgramParameteriEXT(g_ProgramId, GL_GEOMETRY_VERTICES_OUT_EXT, 3);
        //glProgramParameteriEXT(g_ProgramId, GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
        //glProgramParameteriEXT(g_ProgramId, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    }
    if (g_UseFragmentShader)
    {
        // Create shader objects for fragment shader
        GLuint id = pgr2CreateShaderFromFile(GL_FRAGMENT_SHADER, FS_FILE_NAME);
        glAttachShader(g_ProgramId, id);
        glDeleteShader(id);
    }

    // Link shader program
    glLinkProgram(g_ProgramId);
    if (!pgr2CheckProgramLinkStatus(g_ProgramId))
    {
        pgr2CheckProgramInfoLog(g_ProgramId);
        printf("Shader program creation failed.\n\n");
        glDeleteProgram(g_ProgramId);
        g_ProgramId  = 0;
        g_UseShaders = false;
        return;
    }
    else
    {
        printf("Shader program compiled successfully.\n\n");
    }
}


//-----------------------------------------------------------------------------
// Name: initGUI()
// Desc: 
//-----------------------------------------------------------------------------
void initGUI()
{
#ifdef USE_ANTTWEAKBAR
    // Initialize AntTweakBar GUI
    if (!TwInit(TW_OPENGL, NULL))
    {
        assert(0);
    }

    TwWindowSize(g_WindowWidth, g_WindowHeight);
    TwBar *controlBar = TwNewBar("Controls");
    TwDefine(" Controls position='10 10' size='200 360' refresh=0.1 ");

//    TwAddVarCB(controlBar, "use_shaders", TW_TYPE_BOOLCPP, cbSetShaderStatus, cbGetShaderStatus, NULL, " label='shaders' key=s help='Turn programmable pipeline on/off.' ");

    // Shader panel setup
//    TwAddVarRW(controlBar, "vs", TW_TYPE_BOOLCPP, &g_UseVertexShader, " group='Shaders' label='vertex' key=v help='Toggle vertex shader.' ");
//    TwAddVarRW(controlBar, "gs", TW_TYPE_BOOLCPP, &g_UseGeometryShader, " group='Shaders' label='geometry' key=g help='Toggle geometry shader.' ");
//    TwAddVarRW(controlBar, "fs", TW_TYPE_BOOLCPP, &g_UseFragmentShader, " group='Shaders' label='fragment' key=f help='Toggle fragment shader.' ");
//    TwAddButton(controlBar, "build", cbCompileShaderProgram, NULL, " group='Shaders' label='build' key=b help='Build shader program.' ");
//  TwDefine( " Controls/Shaders readonly=true ");   

    // Render panel setup
    TwAddVarRW(controlBar, "wiremode", TW_TYPE_BOOLCPP, &g_WireMode, " group='Render' label='wire mode' key=w help='Toggle wire mode.' ");
    TwAddVarRW(controlBar, "march", TW_TYPE_BOOLCPP, &g_useMarchingCubes, " group='Render' label='March. Cubes' help='Toggle marching cubes.' ");
    //TwAddVarRW(controlBar, "points", TW_TYPE_BOOLCPP, &g_drawPoints, " group='Render' label='Points' help='Draw points.' ");
    TwAddVarRW(controlBar, "melt", TW_TYPE_BOOLCPP, &g_melt, " group='Render' label='Melt' help='Start melting' ");
  
    // Scene panel setup
    TwAddVarRW(controlBar, "auto-rotation", TW_TYPE_BOOLCPP, &g_SceneRotEnabled, " group='Scene' label='rotation' key=r help='Toggle scene rotation.' ");
    TwAddVarRW(controlBar, "Translate", TW_TYPE_FLOAT, &g_SceneTraZ, " group='Scene' label='translate' min=1 max=1000 step=0.5 keyIncr=t keyDecr=T help='Scene translation.' ");
    TwAddVarRW(controlBar, "SceneRotation", TW_TYPE_QUAT4F, &g_SceneRot, " group='Scene' label='rotation' open help='Toggle scene orientation.' ");
#endif
}


//-----------------------------------------------------------------------------
// Name: cbWindowSizeChanged()
// Desc: 
//-----------------------------------------------------------------------------
void cbWindowSizeChanged(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(55.0f, GLfloat(width)/height, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);

    g_WindowWidth  = width;
    g_WindowHeight = height;
}


//-----------------------------------------------------------------------------
// Name: cbKeyboardChanged()
// Desc: 
//-----------------------------------------------------------------------------
void cbKeyboardChanged(int key, int action)
{
    switch (key)
    {

    case 't' : g_SceneTraZ        += 0.5f;                               break;
    case 'T' : g_SceneTraZ        -= (g_SceneTraZ > 0.5) ? 0.5f : 0.0f;  break;
    case 'r' : g_SceneRotEnabled   = !g_SceneRotEnabled;                 break;
    case 'w' : g_WireMode          = !g_WireMode;                        break;
	case 'm' : g_melt			   = !g_melt;	                         break;
    case 'b' : 
        cbCompileShaderProgram(NULL);
        return;
        break;
    }

    printf("[t/T] g_SceneTraZ         = %f\n", g_SceneTraZ);
    printf("[r]   g_SceneRotEnabled   = %s\n", g_SceneRotEnabled ? "true" : "false");
    printf("[w]   g_WireMode          = %s\n", g_WireMode ? "true" : "false");
    printf("[s]   g_UseShaders        = %s\n", g_UseShaders ? "true" : "false");
    printf("[b]   re-compile shaders\n\n");
}

//-----------------------------------------------------------------------------
// Name: cbSetShaderStatus()
// Desc: 
//-----------------------------------------------------------------------------
void TW_CALL cbSetShaderStatus(const void *value, void *clientData)
{
    g_UseShaders = *(bool*)(value);
    // Try to compile shader program
    if (g_UseShaders)
    {
        cbCompileShaderProgram(NULL);
    }
//  TwDefine((g_UseShaders) ? " Controls/Shaders readonly=false " : " Controls/Shaders readonly=true "); 
}


//-----------------------------------------------------------------------------
// Name: cbGetShaderStatus()
// Desc: 
//-----------------------------------------------------------------------------
void TW_CALL cbGetShaderStatus(void *value, void *clientData)
{
    *(bool*)(value) = g_UseShaders;
} 

//-----------------------------------------------------------------------------
// Name: main()
// Desc: 
//-----------------------------------------------------------------------------
int main(int argc, char* argv[]) 
{
	char * s;
	if(COMPUTE_ON_GPU)
		s = "[39GPU] Ice melting - GPU version";
	else
		s = "[39GPU] Ice melting - CPU version";

    return common_main(g_WindowWidth, g_WindowHeight,
                       s,
                       cbInitGL,              // init GL callback function
                       cbDisplay,             // display callback function
                       cbWindowSizeChanged,   // window resize callback function
                       cbKeyboardChanged,     // keyboard callback function
                       NULL,                  // mouse button callback function
                       NULL                   // mouse motion callback function
                       );
}
