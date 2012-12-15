#ifndef __GPUMARCHINGCUBES_H__
#define __GPUMARCHINGCUBES_H__

#include "voxel.h"

class GPUMarchingCubes {

public:
	GPUMarchingCubes();
	~GPUMarchingCubes();
	void vMarchingCubes(Voxel * data,  const int dataCount);

private:
	//Voxel* data;
};


#endif