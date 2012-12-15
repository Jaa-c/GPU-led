#include "cudaCommon.cuh"

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

__device__ float ambientHeat(Voxel * data, int ivoxel) {
	Voxel * voxel = &data[ivoxel];
	return TIME_STEP * (
		(THERMAL_CONDUCTIVITY * (AIR_TEMPERATURE - voxel->temperature))
		/ (SPECIFIC_HEAT_CAP_ICE * voxel->mass)
		);
}

__device__ float transferHeat(Voxel* data, int ivoxel, int iv) {
	Voxel * voxel = &data[ivoxel];
	Voxel * v = &data[iv];
	if(voxel->status == ICE)
		return TIME_STEP * (THERMAL_DIFFUSION_ICE * v->mass * (v->temperature - voxel->temperature) / DENSITY_ICE);
	else if(voxel->status == WATER)
		return TIME_STEP * (THERMAL_DIFFUSION_WATER * v->mass * (v->temperature - voxel->temperature) / DENSITY_WATER);
	else
		return 0;
}


__device__ void updateVoxel(bool condition, Voxel* data, int ivoxel, int iv) {
	Voxel * voxel = &data[ivoxel];
	Voxel * v = &data[iv];
	if(condition) {
		if(v->status != ICE)
			voxel->temperature += ambientHeat(data, ivoxel);
		else {
			float change = transferHeat(data, ivoxel, iv);
			v->temperature += change;
			voxel->temperature -= change;
		}
	}
	else {
		voxel->temperature += ambientHeat(data, ivoxel);
	}
}

__global__ void updateParticlesKernel(Voxel * data) {
	
	const unsigned long threadId = getThreadId();

	if(threadId >= DATA_SIZE)
		return;

	Voxel * voxel = &data[threadId];
	if(voxel->status != ICE) {
		return; //?
	}

	int k = threadId / (WIDTH*HEIGHT);
	int j = (threadId - (k*WIDTH*HEIGHT))/WIDTH;
	int i = threadId - j*WIDTH - k*WIDTH*HEIGHT;

	//okolni castice zjistim podle indexu 
	updateVoxel(i+1 < WIDTH, data, threadId, DATA_INDEX(i+1,j,k));
	updateVoxel(j+1 < HEIGHT, data, threadId, DATA_INDEX(i,j+1,k));
	updateVoxel(k+1 < DEPTH, data, threadId, DATA_INDEX(i,j,k+1));
				
	updateVoxel(i-1 >= 0, data, threadId, DATA_INDEX(i-1,j,k));
	updateVoxel(j-1 >= 0, data, threadId, DATA_INDEX(i,j-1,k));
	updateVoxel(k-1 >= 0, data, threadId, DATA_INDEX(i,j,k-1));
	
	if(voxel->temperature > ZERO_DEG) {
		voxel->status = WATER;
	}
}


__global__ void initDataKernel(Voxel * data) {
	const unsigned long threadId = getThreadId();

	if(threadId > DATA_SIZE)
		return;

	int k = threadId / (WIDTH*HEIGHT);
	int j = (threadId - (k*WIDTH*HEIGHT))/WIDTH;
	int i = threadId - j*WIDTH - k*WIDTH*HEIGHT;

	//shared memory
	float ofsi = 0;//WIDTH/2.0f - 0.5f;
	float ofsj = 0;//HEIGHT/2.0f - 0.5f;
	float ofsk = 0;//DEPTH/2.0f - 0.5f;
	
	Voxel* v = &data[threadId];
	v->position[0] = i - ofsi;
	v->position[1] = j - ofsj;
	v->position[2] = k - ofsk;
		
	if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS)
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
	CHECK_ERR(cudaMemcpy(device_data, host_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyHostToDevice));

	/*dim3 gridRes(32,32,32);
	dim3 blockRes(8,8,8);
	initDataKernel<<< gridRes, blockRes >>>(device_data);
	CHECK_LAST_ERR();
	zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_data, device_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));*/
	
}

void cudaUpdateParticles() {

	dim3 gridRes(32,32,1);
	dim3 blockRes(8,8,8);
	updateParticlesKernel<<< gridRes, blockRes >>>(device_data);
	
	//CHECK_LAST_ERR();

	//cudaThreadSynchronize();
	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_data, device_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));

}

Voxel * cudaGetHostDataPointer() {
	return host_data;
}

void cudaFinalize() {
	CHECK_ERR(cudaFree(&device_data));
}