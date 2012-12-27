//----------------------------------------------------------------------------------------
/**
 * @file       defines.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Contains "global" preprocessor macros.
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __DEFINES_H__
#define __DEFINES_H__

#include "lookup.h"

/** Defines if we use GPU or CPU for computation. true = GPU. */
#define COMPUTE_ON_GPU				true

/**
 * @defgroup PHYSICS Physical constants
 *
 * @{
 */
#define THERMAL_DIFFUSION_WATER		0.1f
#define THERMAL_DIFFUSION_ICE		0.5f
#define DENSITY_ICE					900.0f
#define DENSITY_WATER				1000.0f
#define STEFAN_BOLTZMAN				5.67e-8f
#define THERMAL_CONDUCTIVITY		0.00267f
#define ZERO_DEG					273.15f
#define AIR_TEMPERATURE				(ZERO_DEG + 25.0f)
#define SPECIFIC_HEAT_CAP_ICE		2.11f
/** @} */

/** Defines if all particles have same mass or the mass is slightly different */
#define PARTICLE_RANDOM_MASS		false
/** used if PARTICLE_RANDOM_MASS = true, maximum random value added to particle mass */
#define RAND_MASS_SIZE				(10e-5f)

/** Initial ice temperature */
#define PARTICLE_INIT_TEMPERATURE	(ZERO_DEG - 10.0f)
/** Particle mass */
#define PARTICLE_MASS				0.0008f

/** Time step... */
#define TIME_STEP					0.003f

/** 
 *  @defgroup DATA Size of the grid and ice block
 *
 * @{
 */
#define DATA_WIDTH					(78)
#define DATA_HEIGHT					(48)
#define DATA_DEPTH					(38)

/** Initial air space surrounding ice block */
#define AIR_VOXELS					2

#define WIDTH						(DATA_WIDTH + AIR_VOXELS)
#define HEIGHT						(DATA_HEIGHT + AIR_VOXELS)
#define DEPTH						(DATA_DEPTH + AIR_VOXELS)

#define DATA_SIZE					(WIDTH * HEIGHT * DEPTH)

#define WIDTH_HEIGHT				(WIDTH*HEIGHT)
/** @} */

/** Ice block shape */
#define DATA1		//block
//#define DATA2		//2 connected blocks
//#define DATA3		//block with holes

/** 
 *  @defgroup WINDOW Initial window size
 *
 * @{
 */
#define WINDOW_WIDTH				800
#define WINDOW_HEIGHT				600
/** @} */

/** Echo info into console while melting? */
#define CONSOLE_OUTPUT				true
/** How often output info to console? (in time cycles)  */
#define CONSOLE_OUTPUT_CYCLES		10

/** 3D array indexes -> 1D array index */
#define DATA_INDEX(i,j,k)			((i) + (j)*(WIDTH) + (k)*(WIDTH)*(HEIGHT))

/** 
 *  @defgroup TEST  Correctness test
 *
 * @{
 */
/** Output file with correctness test? */
#define TEST_OUTPUT					true
/** Voxel to output */
#define TEST_VOXEL					DATA_INDEX(20,25,20)
/** @} */

#endif