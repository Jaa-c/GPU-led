//----------------------------------------------------------------------------------------
/**
 * @file       GPUMarchingCubes.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Marching cubes on GPU, only prepared file, algorithm isn't implemented.
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __GPUMARCHINGCUBES_H__
#define __GPUMARCHINGCUBES_H__

#include "voxel.h"

class GPUMarchingCubes {

public:
	GPUMarchingCubes(Voxel * data);
	~GPUMarchingCubes();
	void vMarchingCubes(Voxel * data);

private:
	//Voxel* data;
};


#endif