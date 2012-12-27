//----------------------------------------------------------------------------------------
/**
 * @file       Voxel.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Data structure for a voxel - one unit of the simulation grid.
 *
*/
//----------------------------------------------------------------------------------------

#ifndef __VOXEL_H__
#define __VOXEL_H__

#include "defines.h"
#include "structures.h"

/** Status if the voxel */
enum Status {
	ICE = 0,
	WATER,
	AIR
};

/** Voxel - one unit of the grid */
struct Voxel {
	/** Position in 3D space */
	float position[3];
	/** current teperature of the voxe */
	float temperature;
	/** velocity, not used in this program */
	float velocity;
	/** Mass of the voxel */
	float mass;
	/** Current status of the voxel */
	Status status;

	/** 
	 * Creates new voxel
	 *
	 * @param[in] x X coordinate in space
	 * @param[in] y Y coordinate in space
	 * @param[in] z Z coordinate in space
	 */
	Voxel(const float x, const float y, const float z);

	/** Implicit constructor, does nothing */
	Voxel();

	/** 
	 * Sets voxel position
	 *
	 * @param[in] x X coordinate in space
	 * @param[in] y Y coordinate in space
	 * @param[in] z Z coordinate in space
	 */
	void setPosition(const float x, const float y, const float z);
};

#endif