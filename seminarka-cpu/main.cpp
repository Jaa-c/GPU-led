//----------------------------------------------------------------------------------------
/**
 * @file       main.cpp
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Handles GUI and openGL stuff 
 *
 *  39GPU - seminarni prace - simulace tani ledu na cpu a na gpu
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
/** Vertex shader source file */
const char* VS_FILE_NAME		= "simple.vert";
/** Geometry shader source file */
const char* GS_FILE_NAME		= "simple.geom";
/** Fragment shader source file */
const char* FS_FILE_NAME		= "simple.frag";

/************************************************
*				Default settings			    *
************************************************/

/** Defines if we use GPU or CPU for computation. true = GPU. */
bool	 g_useGPU				= true;
/** handles restart */
bool	 g_initData				= false;

/** Window width */
GLint    g_WindowWidth			= WINDOW_WIDTH;
/** Window height */
GLint    g_WindowHeight			= WINDOW_HEIGHT;

/** Scene orientation */
GLfloat  g_SceneRot[]			= { 0.0f, 0.0f, 0.0f, 1.0f };
/** Scene auto-rotation enabled/disabled */
bool     g_SceneRotEnabled		= false;
/** Wire mode enabled/disabled */
bool     g_WireMode				= false;

/** Use marching cubes algorithm to draw the simulation */
bool     g_useMarchingCubes		= true;
/** To melt or not to melt? */
bool     g_melt					= false;
/** Stores current fps value */
float	 g_fps					= 0;

/** Scene translation along z-axis */
GLfloat  g_SceneTraZ			= std::max(std::max(DATA_WIDTH, DATA_HEIGHT), DATA_DEPTH) * 1.4f;
/** Current data size (number of ICE voxels) */
int realDataCount				= DATA_SIZE;

/** Shader program id */
GLuint g_ProgramId = 0;

/************************************************
*					Simulation				    *
************************************************/
/** The simulation instance */
Simulation* simulation;


/** Current measured time for last n cycles. */
long time_current;
/** Global measured time for last all cycles. */
long time_global;
/* Additional variables storing timestamps for time measuring */
long begin, end, fps_begin, fps_time;
/* Additional variables for object counting */
int cycles, particles, fpsCounter;

/** initialize new data */
bool initMelt = true;

/** File for the test output */
std::ofstream outputTest;


/************************************************
*				Forward declarations		    *
************************************************/
void TW_CALL cbCompileShaderProgram(void *clientData);
void initGUI();

/**
 * Main openGL display loop
 *
 * Handles what happens on each frame.
 */
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

	//count FPS
	fpsCounter++;
	fps_time = timeGetTime() - fps_begin;
	if(fps_time > 1000) {
		g_fps = fpsCounter/(fps_time/1000.0f);
		fps_begin = timeGetTime();
		fpsCounter = 0;
	}

	if(g_initData) { //on restart
		g_initData = false;
		simulation->init();
		time_global = 0;
		time_current = 0;
		cycles = CONSOLE_OUTPUT_CYCLES;
		if(CONSOLE_OUTPUT) {
			std::cout << "\n    # Restarting...\n";
		}
	}

	//melt ice
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
		
		//job well done!
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
				char * s = "GPU";
				if(!g_useGPU)
					s = "CPU";
				std::cout << s << " > ice particles: " << particles << " time: " << time_current << "ms, sum: " << time_global << "ms\n";
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
	
    //programmable pipeline:
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

		/* * /
		GLint hVertex = glGetAttribLocation(g_ProgramId, "a_Position");
		glVertexAttribPointer(hVertex, 3, GL_FLOAT, GL_FALSE, 0, &triangleNet);
		glEnableVertexAttribArray(hVertex);
	
		glDrawArrays(GL_TRIANGLES, 0, triangleNetIndex/3);
	
		glDisableVertexAttribArray(hVertex);
		/**/
	}

    // Turn off programmable pipeline
    glUseProgram(NULL);
    if (g_SceneRotEnabled)
    {
        scene_rot+=0.25;
    }
}

/** Initializes openGL and simulation status */
void cbInitGL()
{
	simulation = new GPUSimulation();
	simulation->init();

	//glEnableVertexAttribArray(1);
	srand ( time(NULL) );
	g_fps = 0;
	fps_begin = timeGetTime();

	if(TEST_OUTPUT) {
		if(g_useGPU)
			outputTest.open ("test-gpu.txt");
		else
			outputTest.open ("test-cpu.txt");
	}
	
    // Init app GUI
    initGUI();

    // Set OpenGL state variables
    glClearColor(0.15f, 0.15f, 0.15f, 0);
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

    cbCompileShaderProgram(NULL);
}


/** Compiles shader programs */
void TW_CALL cbCompileShaderProgram(void *clientData)
{
    // Delete shader program if exists
    if (g_ProgramId)
    {
        glDeleteProgram(g_ProgramId);
    }

    // Create shader program object
    g_ProgramId = glCreateProgram();

    // Create shader objects for vertex shader
    GLuint id = pgr2CreateShaderFromFile(GL_VERTEX_SHADER, VS_FILE_NAME);
    glAttachShader(g_ProgramId, id);
    glDeleteShader(id);

    // Create shader objects for geometry shader
    id = pgr2CreateShaderFromFile(GL_GEOMETRY_SHADER, GS_FILE_NAME);
    glAttachShader(g_ProgramId, id);
    glDeleteShader(id);

    // Create shader objects for fragment shader
    id = pgr2CreateShaderFromFile(GL_FRAGMENT_SHADER, FS_FILE_NAME);
    glAttachShader(g_ProgramId, id);
    glDeleteShader(id);

    // Link shader program
    glLinkProgram(g_ProgramId);
    if (!pgr2CheckProgramLinkStatus(g_ProgramId))
    {
        pgr2CheckProgramInfoLog(g_ProgramId);
        printf("Shader program creation failed.\n\n");
        glDeleteProgram(g_ProgramId);
        g_ProgramId  = 0;
        return;
    }
    else
    {
        printf("Shader program compiled successfully.\n\n");
    }
}


/** Anttweakbar callback for gpu vs cpu button */
void TW_CALL cbSetGPUUsage(const void *value, void *clientData)
{
    if (!g_melt) //change only if not melting
    {
        g_useGPU = !g_useGPU;
    }
	else {
		if(CONSOLE_OUTPUT) {
			std::cout << "    # Stop simulation before changing device..\n";
		}
		return;
	}

	delete simulation;
	g_initData = true;
	if(g_useGPU) {
		simulation = new GPUSimulation();
	}
	else {
		simulation = new CPUSimulation();
	}
}

/** Anttweakbar callback for gpu vs cpu button */
void TW_CALL cbGetGPUUsage(void *value, void *clientData)
{
    *(bool*)(value) = g_useGPU;
} 

/** Initializes GUI - only AntweakBar menu creation */
void initGUI()
{
    // Initialize AntTweakBar GUI
    if (!TwInit(TW_OPENGL, NULL))
    {
        assert(0);
    }

    TwWindowSize(g_WindowWidth, g_WindowHeight);
    TwBar *controlBar = TwNewBar("Controls");
    TwDefine("Controls position='10 10' size='200 300' refresh=0.1  color='130 140 150'");
	
	TwAddVarCB(controlBar, "gpu", TW_TYPE_BOOLCPP, cbSetGPUUsage, cbGetGPUUsage, NULL, " group='CUDA' label='Compute on:' true='GPU' false='CPU' ");
	//TwAddVarRW(controlBar, "restart", TW_TYPE_BOOLCPP, &g_initData, " group='PROGRAM' label='Restart: ' true='' false='restart' ");
    
    TwAddVarRW(controlBar, "wiremode", TW_TYPE_BOOLCPP, &g_WireMode, " group='Render' label='Wire mode' key=w help='Toggle wire mode.' ");
    TwAddVarRW(controlBar, "march", TW_TYPE_BOOLCPP, &g_useMarchingCubes, " group='Render' label='March. Cubes' help='Toggle marching cubes.' ");
    TwAddVarRW(controlBar, "melt", TW_TYPE_BOOLCPP, &g_melt, " group='Render' label='Melt' key=m help='Start melting' ");
  
    TwAddVarRW(controlBar, "auto-rotation", TW_TYPE_BOOLCPP, &g_SceneRotEnabled, " group='Scene' label='rotation' key=r help='Toggle scene rotation.' ");
    TwAddVarRW(controlBar, "Translate", TW_TYPE_FLOAT, &g_SceneTraZ, " group='Scene' label='translate' min=1 max=1000 step=10 keyIncr=t keyDecr=T help='Scene translation.' ");
    TwAddVarRW(controlBar, "SceneRotation", TW_TYPE_QUAT4F, &g_SceneRot, " group='Scene' label='rotation' open help='Toggle scene orientation.' ");

	TwAddVarRO(controlBar, "fps", TW_TYPE_FLOAT, &g_fps, " group='FPS' label='Current fps' ");
}

/**
 * This method handles changes of window size
 * 
 * @param[in] width New window width
 * @param[in] height New window height
 */
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

/** Handles keybord action */
void cbKeyboardChanged(int key, int action)
{
    switch (key)
    {
    case 't' : g_SceneTraZ        += 0.5f;                               break;
    case 'T' : g_SceneTraZ        -= (g_SceneTraZ > 0.5) ? 0.5f : 0.0f;  break;
    case 'r' : g_SceneRotEnabled   = !g_SceneRotEnabled;                 break;
    case 'w' : g_WireMode          = !g_WireMode;                        break;
	case 'm' : g_melt			   = !g_melt;	                         break;
    }

    printf("[t/T] g_SceneTraZ         = %f\n", g_SceneTraZ);
    printf("[r]   g_SceneRotEnabled   = %s\n", g_SceneRotEnabled ? "true" : "false");
    printf("[w]   g_WireMode          = %s\n", g_WireMode ? "true" : "false");
    printf("[m]   g_melt              = %s\n", g_melt ? "true" : "false");
}


/** The main. */
int main(int argc, char* argv[]) 
{
    return common_main(g_WindowWidth, g_WindowHeight,
                       "[39GPU] Ice melting simulation",
                       cbInitGL,              // init GL callback function
                       cbDisplay,             // display callback function
                       cbWindowSizeChanged,   // window resize callback function
                       cbKeyboardChanged,     // keyboard callback function
                       NULL,                  // mouse button callback function
                       NULL                   // mouse motion callback function
                       );
}
