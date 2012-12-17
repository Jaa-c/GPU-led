#ifndef __CPUMARCHINGCUBES_H__
#define __CPUMARCHINGCUBES_H__

#include "voxel.h"

class CPUMarchingCubes {

public:
	CPUMarchingCubes();
	~CPUMarchingCubes();
	void vMarchingCubes(Voxel * data);

private:
	void vMarchCube(const int fX, const  int fY, const int fZ, const GLfloat fScale = 1.0f);
	Voxel* data;
};


#endif