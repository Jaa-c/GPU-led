//----------------------------------------------------------------------------------------
/**
 * @file       CPUMarchingCubes.h
 * @author     Daniel Princ
 * @date       2012/12/13
 * @brief	   Marching cubes implementation on the GPU
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
 *
 * This class has no buffer output, it produces triangles via glVertex3f().
 * (That is not the best solution, should use VertexBufferObject, but that
 * requires to pre-compute the number of triangles produced by marching cubes.
 * That is quite complex problem.)
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
	 * Does the marching cubes algorithm on one voxel, generates 0-5 triangles with glVertex3f().
	 *
	 * @param[in] fX X coordinate in the grid of current vocel
	 * @param[in] fY Y coordinate in the grid of current vocel
	 * @param[in] fZ Z coordinate in the grid of current vocel
	 */
	void vMarchCube(const int fX, const  int fY, const int fZ);
	/** Pointer to current grid, that is used to create surface with marching cubes. */
	const Voxel* data;
};

#endif