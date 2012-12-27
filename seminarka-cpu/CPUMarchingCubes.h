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