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
	void updateVoxel(bool condition, Voxel* voxel, Voxel* v);
	float transferHeat(Voxel * voxel, Voxel* v);
	float ambientHeat(Voxel *voxel);

	CPUMarchingCubes* cpumc;
};

#endif