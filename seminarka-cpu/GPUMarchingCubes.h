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