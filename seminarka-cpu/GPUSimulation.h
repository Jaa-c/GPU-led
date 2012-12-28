//----------------------------------------------------------------------------------------
/**
 * @file       GPUSimulation.h
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Simulation on the GPU
 *
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __GPUSIMULATION_H__
#define __GPUSIMULATION_H__

#include "Simulation.h"
#include "CPUMarchingCubes.h"

class GPUSimulation : public Simulation {

public:
	/** 
	 * Implicit construtor allocates both buffers for the grid and 
	 * creates new instance of marching cubes. 
	 */
	GPUSimulation();
	/** Implicit destrutor frees allocated memory. */
	~GPUSimulation();
	virtual Voxel* getData();
	virtual int updateParticles();
	virtual void init();
	virtual void march();

	virtual void setData(Voxel * data);

private:
	/** Pointer to marching cubes instance */
	CPUMarchingCubes* cpumc;
};

#endif