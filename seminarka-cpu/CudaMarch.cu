//----------------------------------------------------------------------------------------
/**
 * @file       CudaMarch.cuh
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Prepared file for marching cubes on GPU, not implemented
 * 
 * This file is preparation for marching cubes in cuda,
 * however it is not implemented.
 *
*/
//----------------------------------------------------------------------------------------

#include "cudaCommon.cuh"
#include "GPUComputation.cuh"

Voxel * device_data_pointer = 0;

__device__ unsigned long getThreadID() {
	return NULL;
}


__global__ void marchCubes(Voxel * data, float * verticies, int count) {

}

float * vertex_data;
int vertex_count;
void cudaMarchInit(Voxel * host_data) {
	device_data_pointer = cudaGetDeviceDataPointer();
	//host_data_pointer = device_data;

	CHECK_ERR(cudaMalloc((void**)&vertex_data, DATA_SIZE * 3 *  sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&vertex_count, sizeof(int)));
}


void cudaMarchingCubes() {


}