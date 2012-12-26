#ifndef __DEFINES_H__
#define __DEFINES_H__

#include "lookup.h"

#define COMPUTE_ON_GPU				true

#define THERMAL_DIFFUSION_WATER		0.1f
#define THERMAL_DIFFUSION_ICE		0.5f
#define DENSITY_ICE					900.0f
#define DENSITY_WATER				1000.0f
#define STEFAN_BOLTZMAN				5.67e-8f
#define THERMAL_CONDUCTIVITY		0.00267f
#define ZERO_DEG					273.15f
#define AIR_TEMPERATURE				(ZERO_DEG + 25.0f)
#define SPECIFIC_HEAT_CAP_ICE		2.11f

#define PARTICLE_RANDOM_MASS		false
#define RAND_MASS_SIZE				(10e-5f)

#define PARTICLE_INIT_TEMPERATURE	(ZERO_DEG - 10.0f)
#define PARTICLE_MASS				0.0008f

#define TIME_STEP					0.003f

#define DATA_WIDTH					(78)
#define DATA_HEIGHT					(48)
#define DATA_DEPTH					(38)

#define AIR_VOXELS					2

#define WIDTH						(DATA_WIDTH + AIR_VOXELS)
#define HEIGHT						(DATA_HEIGHT + AIR_VOXELS)
#define DEPTH						(DATA_DEPTH + AIR_VOXELS)

#define DATA_SIZE					(WIDTH * HEIGHT * DEPTH)

#define WIDTH_HEIGHT				(WIDTH*HEIGHT)

#define WINDOW_WIDTH				800
#define WINDOW_HEIGHT				600

#define CONSOLE_OUTPUT				true
#define CONSOLE_OUTPUT_CYCLES		10

#define DATA_INDEX(i,j,k)			((i) + (j)*(WIDTH) + (k)*(WIDTH)*(HEIGHT))

#define TEST_OUTPUT					true

#endif