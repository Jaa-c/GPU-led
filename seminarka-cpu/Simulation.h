//----------------------------------------------------------------------------------------
/**
 * @file       Simulation.h
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Abstract class for simulation.
 *
 */
//----------------------------------------------------------------------------------------

#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include <algorithm>
#include "voxel.h"

class Simulation {
protected:
	/** Buffer in which the next state of the simulation grid is written */
	Voxel * writeData;
	/** Buffer that contains the current state of the simulation grid */
	Voxel * readData;
	/** Number of ice particles */
	int iceParticles;

public:
	/** 
	 * Returns the poiter to current data 
	 *
	 * @return Poiter to current data 
	 */
	virtual Voxel* getData() = 0;
	/** 
	 * Updates the whole grid
	 *
	 * @return number of particles that have melted
	 */
	virtual int updateParticles() = 0;
	/** Initializes the data to the default values */
	virtual void init() = 0;
	/** Calls the marching cubes algorithm and draws current ice surface */
	virtual void march() = 0;
	/**
	 * Sets the grid. Debug purposes only.
	 *
	 * @param[in] data The grid.
	 */
	virtual void setData(Voxel * data) = 0;

};

#endif