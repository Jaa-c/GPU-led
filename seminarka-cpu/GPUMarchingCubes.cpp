#include "GPUMarchingCubes.h"
#include "GPUComputation.cuh"

GPUMarchingCubes::GPUMarchingCubes(Voxel * data) {
	cudaMarchInit(data);
}

void GPUMarchingCubes::vMarchingCubes(Voxel * data) {

}