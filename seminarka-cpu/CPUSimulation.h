//----------------------------------------------------------------------------------------
/**
 * @file       CPUSimulation.h
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Proveds simulation on the CPU
 *
 *  Simulation on the CPU
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __CPUSIMULATION_H__
#define __CPUSIMULATION_H__

#include "Simulation.h"
#include "CPUMarchingCubes.h"

/**
 * Class, that does the simulation on the CPU.
 *
 * It provides methods to update the grid in each iteration
 * and to draw the grid with marching cubes. 
 */
class CPUSimulation : public Simulation {

public:
	/** 
	 * Implicit construtor allocates both buffers for the grid and 
	 * creates new instance of marching cubes. 
	 */
	CPUSimulation();
	/** Implicit destrutor frees allocated memory. */
	~CPUSimulation();
	virtual Voxel* getData();
	virtual int updateParticles();
	virtual void init();
	virtual void march();
	
	virtual void setData(Voxel * data);

private:
	/**
	 * Updates temperature of the given voxel based on 1 neighbouring particle
	 * 
	 * @param[in] condition Condition, whether the neighbouring particle exists (isn't out of grid).
	 * @param[in,out] writeVoxel Pointer to current voxel from the write buffer
	 * @param[in,out] writeV Pointer to neighbouring voxel from the write buffer
	 * @param[in] readVoxel Pointer to current voxel from the read buffer
	 * @param[in] readV Pointer to neighbouring voxel from the read buffer
	 */
	void updateVoxel(const bool condition, Voxel * writeVoxel,  Voxel * writeV , const Voxel* readVoxel, const Voxel* readV);
	/**
	 * Computes transffered heat from one voxel to the other
	 * 
	 * @param[in] voxel Pointer to current voxel
	 * @param[in] v Pointer to neighbouring voxel
	 *
	 * @return The amount of heat to transfer
	 */
	float transferHeat(const Voxel * voxel, const Voxel* v);
	/**
	 * Computes ambient heat, that voxel gets from surrounding air
	 * 
	 * @param[in] voxel Pointer to current voxel
	 *
	 * @return The amount of heat to transfer
	 */
	float ambientHeat(const Voxel *voxel);

	/** Pointer to marching cubes instance */
	CPUMarchingCubes* cpumc;
};

#endif