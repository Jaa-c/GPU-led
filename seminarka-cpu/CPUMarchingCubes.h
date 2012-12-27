//----------------------------------------------------------------------------------------
/**
 * @file       CPUMarchingCubes.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Marching cubes algorithm implemented on the CPU.
 *
 */
//----------------------------------------------------------------------------------------


#ifndef __CPUMARCHINGCUBES_H__
#define __CPUMARCHINGCUBES_H__

#include "voxel.h"

/**
 * A class reprezenting marching cubes algorithm, implemented on the CPU.
 */
class CPUMarchingCubes {

public:
	/** Implicit constructor, does nothing */
	CPUMarchingCubes();
	/** Implicit destructor, does nothing */
	~CPUMarchingCubes();
	/**
	 * Does the marching cubes algorithm, draws the surface on the 
	 * edge between ice and air/water.
	 *
	 * @param[in] data pointer to the simulation grid
	 */
	void vMarchingCubes(const Voxel * data);

private:
	/**
	 * Does the marching cubes algorithm on one voxel
	 *
	 * @param[in] fX X coordinate in the grid of current vocel
	 * @param[in] fY Y coordinate in the grid of current vocel
	 * @param[in] fZ Z coordinate in the grid of current vocel
	 */
	void vMarchCube(const int fX, const  int fY, const int fZ);
	/** pointer to current grid */
	const Voxel* data;
};

#endif