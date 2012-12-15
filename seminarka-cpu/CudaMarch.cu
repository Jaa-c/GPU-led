#include "cudaCommon.cuh"
#include "GPUComputation.cuh"

Voxel * host_data_pointer = 0;

__global__ void marchCubes() {




}


void cudaMarchInit() {
	host_data_pointer = cudaGetHostDataPointer();
}


void cudaMarchingCubes() {


}