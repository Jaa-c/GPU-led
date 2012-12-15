#include "GPUMarchingCubes.h"
#include "GPUComputation.cuh"

GPUMarchingCubes::GPUMarchingCubes() {
	cudaMarchInit();
}

void GPUMarchingCubes::vMarchingCubes(Voxel * data,  const int dataCount) {

}