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

	virtual void setData(Voxel * data);

private:
	int dataWidth, dataHeight, dataDepth;
	int dataCount;
	CPUMarchingCubes* cpumc;
};

#endif