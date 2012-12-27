//----------------------------------------------------------------------------------------
/**
 * @file       CPUSimulation.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Simulation on the CPU
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __CPUSIMULATION_H__
#define __CPUSIMULATION_H__

#include "Simulation.h"
#include "CPUMarchingCubes.h"

class CPUSimulation : public Simulation {

public:
	CPUSimulation();
	~CPUSimulation();
	virtual Voxel* getData();
	virtual int updateParticles();
	virtual void init();
	virtual void march();
	
	virtual void setData(Voxel * data);

private:
	void updateVoxel(bool condition, Voxel * writeVoxel,  Voxel * writeV , Voxel* readVoxel, Voxel* readV);
	float transferHeat(Voxel * voxel, Voxel* v);
	float ambientHeat(Voxel *voxel);

	CPUMarchingCubes* cpumc;
};

#endif