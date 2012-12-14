#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include "voxel.h"

class Simulation {
protected:
	Voxel * data;

public:
	virtual Voxel* getData() = 0;
	virtual void updateParticles() = 0;
	virtual void init() = 0;
	virtual void march() = 0;

};

#endif