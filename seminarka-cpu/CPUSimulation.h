#ifndef __CPUSIMULATION_H__
#define __CPUSIMULATION_H__

#include "Simulation.h"
#include "CPUMarchingCubes.h"

class CPUSimulation : public Simulation {

public:
	CPUSimulation(int dataWidth, int dataHeight, int dataDepth);
	~CPUSimulation();
	virtual Voxel* getData();
	virtual void updateParticles();
	virtual void init();
	virtual void march();
	
	virtual void setData(Voxel * data);

private:
	void updateVoxel(bool condition, Voxel* voxel, Voxel* v);
	float transferHeat(Voxel * voxel, Voxel* v);
	float ambientHeat(Voxel *voxel);

	int dataWidth, dataHeight, dataDepth;
	int dataCount;
	CPUMarchingCubes* cpumc;
};

#endif