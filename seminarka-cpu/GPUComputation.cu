
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Voxel.h"

inline void HandleError( cudaError_t error, const char *file, int line )
{
	if (error != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
		//exit( EXIT_FAILURE );
	}
}
#define CHECK_ERR( error ) HandleError( error, __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line ) {
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
	}

	err = cudaDeviceSynchronize();
	if( cudaSuccess != err ) {
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
	}
}
#define CHECK_LAST_ERR() __cudaCheckError( __FILE__, __LINE__ )

/************************************************
*				Device buffers			    *
************************************************/
Voxel * device_data = 0;
Voxel * host_data = 0;

__device__ unsigned long getThreadId() {
	const unsigned long block = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	
	return  block * blockDim.x * blockDim.y * blockDim.z +
			threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
}

__device__ float ambientHeat(Voxel *voxel) {
	return TIME_STEP * (
		(THERMAL_CONDUCTIVITY * (AIR_TEMPERATURE - voxel->temperature))
		/ (SPECIFIC_HEAT_CAP_ICE * voxel->mass)
		);
}

__device__ float transferHeat(Voxel * voxel, Voxel* v) {
	if(voxel->status == ICE)
		return TIME_STEP * (THERMAL_DIFFUSION_ICE * v->mass * (v->temperature - voxel->temperature) / DENSITY_ICE);
	else if(voxel->status == WATER)
		return TIME_STEP * (THERMAL_DIFFUSION_WATER * v->mass * (v->temperature - voxel->temperature) / DENSITY_WATER);
	else
		return 0;
}


__device__ void updateVoxel(bool condition, Voxel* data, int v1, int v2) {
	Voxel * voxel = &data[v1];
	//Voxel * v = &data[v2];

	if(condition ) {//&& v->status != ICE) {
		//float change = transferHeat(voxel, v);
		float change = 1.f;
		voxel->temperature -= change;
		//v->temperature += change;
	}
	else {
		voxel->temperature += 1.f;//ambientHeat(voxel);
	}
}

__global__ void updateParticlesKernel(Voxel * data) {
	const unsigned long block = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	
	const unsigned long threadId = block * blockDim.x * blockDim.y * blockDim.z +
			threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

	if(threadId >= DATA_SIZE)//=?
		return;

	Voxel * voxel = &data[threadId];

	if(voxel->status != ICE) {
		return; //?
	}
	//do sdileny pameti (konstatntni?)
	const int neighbours[6][3] = {
		{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 
		{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}
	};

	int k = threadId/DATA_WIDTH_TOTAL/DATA_HEIGHT_TOTAL;
	int j = ((threadId - k*DATA_WIDTH_TOTAL*DATA_HEIGHT_TOTAL)/DATA_WIDTH_TOTAL) % DATA_HEIGHT_TOTAL;
	int i = threadId - j*DATA_WIDTH_TOTAL - k*DATA_WIDTH_TOTAL*DATA_HEIGHT_TOTAL;
	
	//okolni castice zjistim podle indexu 
	updateVoxel(i+neighbours[0][0] < DATA_WIDTH_TOTAL, data, threadId, DATA_INDEX(i+neighbours[0][0],j,k));
	updateVoxel(j+neighbours[1][1] < DATA_HEIGHT_TOTAL, data, threadId, DATA_INDEX(i,j+neighbours[1][1],k));
	updateVoxel(k+neighbours[2][2] < DATA_DEPTH_TOTAL, data, threadId, DATA_INDEX(i,j,k+neighbours[2][2]));
				
	updateVoxel(i+neighbours[3][0] >= 0, data, threadId, DATA_INDEX(i+neighbours[3][0],j,k));
	updateVoxel(j+neighbours[4][1] >= 0, data, threadId, DATA_INDEX(i,j+neighbours[4][1],k));
	updateVoxel(k+neighbours[5][2] >= 0, data, threadId, DATA_INDEX(i,j,k+neighbours[5][2]));
	
	if(voxel->temperature > ZERO_DEG) {
		voxel->status = WATER;
	}
}



//deprecated... ono je to vygeneruje ve spanym poradi... (logicky)
__global__ void initDataKernel(Voxel * data) {
	const unsigned long threadId = getThreadId();

	if(threadId > DATA_SIZE)
		return;

	int k = threadId/DATA_WIDTH_TOTAL/DATA_HEIGHT_TOTAL;
	int j = ((threadId - k*DATA_WIDTH_TOTAL*DATA_HEIGHT_TOTAL)/DATA_WIDTH_TOTAL) % DATA_HEIGHT_TOTAL;
	int i = threadId - j*DATA_WIDTH_TOTAL - k*DATA_WIDTH_TOTAL*DATA_HEIGHT_TOTAL;
	
	Voxel* v = &data[threadId];
	v->position[0] = i;
	v->position[1] = j;
	v->position[2] = k;
	
	if(i <= 1 || j <= 1 || k <= 1)
		v->status = AIR; //nastavim maly okoli na vzduch

}


void cudaInit(Voxel * data) {
	//choosing the best CUDA device
	//code fragment taken from https://www.cs.virginia.edu/~csadmin/wiki/index.php/CUDA_Support/Choosing_a_GPU
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1) {
		int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount) {
					max_multiprocessors = properties.multiProcessorCount;
					max_device = device;
			}
		}
		CHECK_ERR(cudaSetDevice(max_device));
	}
	else {
		CHECK_ERR(cudaSetDevice(0));
	}

	host_data = data;

	//alocate host data
	CHECK_ERR(cudaMalloc((void**)&device_data, DATA_SIZE * sizeof(Voxel)));

	//copy data from host to device
	CHECK_ERR(cudaMemcpy(device_data, host_data, DATA_SIZE *  sizeof(Voxel), cudaMemcpyHostToDevice));
	
}

void cudaUpdateParticles() {

	dim3 gridRes(32,32,1);
	dim3 blockRes(8,8,8);
	updateParticlesKernel<<< gridRes, blockRes >>>(device_data);
	
	CHECK_LAST_ERR();

	cudaThreadSynchronize();
	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_data, device_data, DATA_SIZE *  sizeof(Voxel), cudaMemcpyDeviceToHost));

}


void cudaFinalize() {
	CHECK_ERR(cudaFree(&device_data));
}