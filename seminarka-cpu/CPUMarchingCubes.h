#ifndef __CPUMARCHINGCUBES_H__
#define __CPUMARCHINGCUBES_H__

#include "voxel.h"

class CPUMarchingCubes {

public:
	CPUMarchingCubes(const int dataWidth, const int dataHeight, const int dataDepth);
	~CPUMarchingCubes();
	void vMarchingCubes(Voxel * data,  const int dataCount);

private:
	void vMarchCube(const int fX, const  int fY, const int fZ, const int dataCount, const GLfloat fScale = 1.0f);
	int dataWidth, dataHeight, dataDepth;
	Voxel* data;
};


#endif