#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include "voxel.h"

class Simulation {
protected:
	Voxel * data;
	int iceParticles;
	void initData();

public:
	virtual Voxel* getData() = 0;
	virtual int updateParticles() = 0;
	virtual void init() = 0;
	virtual void march() = 0;

	virtual void setData(Voxel * data) = 0;

};

#endif