//----------------------------------------------------------------------------------------
/**
 * @file       voxel.h
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Data structure for a voxel - one unit of the simulation grid.
 *
*/
//----------------------------------------------------------------------------------------

#ifndef __VOXEL_H__
#define __VOXEL_H__

#include "defines.h"
#include "structures.h"

/** State of the voxel */
enum Status {
	ICE = 0,
	WATER,
	AIR
};

/**
 * Voxel is a basic data structure, that represents one 
 * unit in the simulation grid.
 */
struct Voxel {
	/** Position of the unit in 3D space */
	float position[3];
	/** Current teperature of the voxel */
	float temperature;
	/** Velocity, currently not used in this program */
	float velocity;
	/** Mass of the voxel */
	float mass;
	/** Current status of the voxel based on the tempearture */
	Status status;

	/** 
	 * Creates new voxel in a given location
	 *
	 * @param[in] x X coordinate in space
	 * @param[in] y Y coordinate in space
	 * @param[in] z Z coordinate in space
	 */
	Voxel(const float x, const float y, const float z);

	/** Implicit constructor, does nothing */
	Voxel();

	/** 
	 * Sets voxel position in 3D space
	 *
	 * @param[in] x X coordinate in space
	 * @param[in] y Y coordinate in space
	 * @param[in] z Z coordinate in space
	 */
	void setPosition(const float x, const float y, const float z);
};

#endif