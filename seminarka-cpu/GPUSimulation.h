#ifndef __GPUSIMULATION_H__
#define __GPUSIMULATION_H__

#include "Simulation.h"
#include "CPUMarchingCubes.h"

class GPUSimulation : public Simulation {

public:
	GPUSimulation(int dataWidth, int dataHeight, int dataDepth);
	~GPUSimulation();
	virtual Voxel* getData();
	virtual void updateParticles();
	virtual void init();
	virtual void march();

private:
	void updateVoxel(bool condition, Voxel* voxel, Voxel* v);
	float transferHeat(Voxel * voxel, Voxel* v);
	float ambientHeat(Voxel *voxel);

	int dataWidth, dataHeight, dataDepth;
	int dataCount;
	CPUMarchingCubes* cpumc;
};

#endif