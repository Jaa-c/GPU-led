//----------------------------------------------------------------------------------------
/**
 * @file       GPUComputation.cuh
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Header file for cuda methods. 
 *
*/
//----------------------------------------------------------------------------------------

#ifndef __GPUCOMPUTATION_CUH__
#define __GPUCOMPUTATION_CUH__

/**
 * Initializes cuda, chooses cuda device, allocates device memory etc.
 * Also initializes the default grid.
 *
 * @param[in] readData Pointer to read buffer in host memory
 * @param[in,out] writeData Pointer to write buffer in host memory
 * @param[out] host_ice Pointer to number of ice voxels(status=ICE) in host memory
 */
void cudaInit(Voxel * readData, Voxel * writeData, int * host_ice);

/** Cleanup, frees device memory */
void cudaFinalize();

/**
 * Updates the whole grid
 * 
 * @param[out] ice Number of voxels that melted in this iteration
 */
void cudaUpdateParticles(int * ice);

/**
 * @return The pointer to the grid in device memory
 */
Voxel * cudaGetDeviceDataPointer();

/**
 * Initializes marching cubes. NOT IMPLEMENTED
 *
 * @param[in] host_data pointer to device data
 */
void cudaMarchInit(Voxel * device_data);

/**
 * Marching cubes in cuda. NOT IMPLEMETED
 */
void cudaMarchingCubes();

#endif