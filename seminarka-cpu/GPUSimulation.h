//----------------------------------------------------------------------------------------
/**
 * @file       GPUSimulation.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Simulation on the GPU
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __GPUSIMULATION_H__
#define __GPUSIMULATION_H__

#include "Simulation.h"
#include "CPUMarchingCubes.h"

class GPUSimulation : public Simulation {

public:
	GPUSimulation();
	~GPUSimulation();
	virtual Voxel* getData();
	virtual int updateParticles();
	virtual void init();
	virtual void march();

	virtual void setData(Voxel * data);

private:
	CPUMarchingCubes* cpumc;
};

#endif