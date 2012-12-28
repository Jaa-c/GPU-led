//----------------------------------------------------------------------------------------
/**
 * @file       GPUMarchingCubes.h
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Marching cubes on GPU, only prepared file, algorithm isn't implemented.
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __GPUMARCHINGCUBES_H__
#define __GPUMARCHINGCUBES_H__

#include "voxel.h"

/**
 * This is a prepared class for marching cubes implementation on GPU.
 * The algorithm isn't implemented. (To be done in the future :)
 */
class GPUMarchingCubes {

public:
	GPUMarchingCubes(Voxel * data);
	~GPUMarchingCubes();
	void vMarchingCubes(Voxel * data);

private:
	//Voxel* data;
};


#endif