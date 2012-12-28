//----------------------------------------------------------------------------------------
/**
 * @file       CudaUpdate.cu
 * @author     Daniel Princ
 * @date       2012/12/13 
 * @brief	   Main cuda file, updates the grid on GPU
 *
*/
//----------------------------------------------------------------------------------------

#include "cudaCommon.cuh"

/************************************************
*			  Pointers to the data			    *
************************************************/
/** Pointer to write buffer in the device memory */
Voxel * device_write_data = 0;
/** Pointer to read buffer in the device memory */
Voxel * device_read_data = 0;
/** Pointer to write buffer in the host memory */
Voxel * host_write_data = 0;
/** Pointer to read buffer in the host memory */
Voxel * host_read_data = 0;

/** Pointer to sum of melted voxels in the device memory */
int * device_ice_data = 0;

/************************************************
*		     Grid and block size			    *
************************************************/
/** Number of threads in block */
#define	BLOCK_THREADS	512
/** Grid resolution, based on data size */
#define	GRID_RES		((int) (pow(DATA_SIZE / BLOCK_THREADS, 1.0f / 3.0f) + 1.0f))
/** Size of the grid */
#define GRID_SIZE		(GRID_RES*GRID_RES*GRID_RES)

/** Cuda block resolution */
const dim3 blockRes(8,8,8);
/** Cuda grid resolution */
const dim3 gridRes(GRID_RES, GRID_RES, GRID_RES);

/************************************************
*				Memory declaration			    *
************************************************/
/** Cache for parallel reduction */
__shared__ int cache[BLOCK_THREADS];
/** Offset that moves the the middle of the ice block to 0,0,0 coordinates */
__constant__ float  positionOffset[3] = { WIDTH/2.0f - 0.5f, HEIGHT/2.0f - 0.5f, DEPTH/2.0f - 0.5f};
/** Lookup table for thermal diffusion constant */
__constant__ float thermal_diffusion[3] = {THERMAL_DIFFUSION_ICE, THERMAL_DIFFUSION_WATER, 1.0f};
/** Lookup table for material density */
__constant__ float density[3] = {DENSITY_ICE, DENSITY_WATER, 1.0f};
/** Lookup table that determines if the heat is transferred or nor (based od voxel state) */
__constant__ int transfer[3] = {1, 1, 0};


/**
 * Computes ambient heat, that voxel gets from surrounding air
 * 
 * @param[in] data Pointer to device data
 * @param[in] ivoxel Index in the data to the current voxel
 * @return The amount of heat to transfer
 */
__device__ __forceinline__ float ambientHeat(const Voxel * data, const int ivoxel) {
	return TIME_STEP * (
		(THERMAL_CONDUCTIVITY * (AIR_TEMPERATURE - (&data[ivoxel])->temperature))
		/ (SPECIFIC_HEAT_CAP_ICE * (&data[ivoxel])->mass)
		);
}

/**
 * Computes transffered heat from one voxel to the other
 * 
 * @param[in] data Pointer to device data
 * @param[in] ivoxel Index in the data to the current voxel
 * @param[in] iv Index in the data to the neighbouring voxel
 *
 * @return The amount of heat to transfer
 */
__device__ __forceinline__ float transferHeat(const Voxel * data, const int ivoxel, const int iv) {

	return transfer[(&data[ivoxel])->status] * 
		(TIME_STEP * (thermal_diffusion[(&data[ivoxel])->status] * 
		(&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / density[(&data[ivoxel])->status]));
}

/**
 * Updates temperature of the given voxel based on neighbouring particle
 * 
 * @param[in] condition Condition, whether the neighbouring particle exists (isn't out of grid)
 * @param[in] readData Pointer to device read buffer
 * @param[in] writeData Pointer to device write buffer
 * @param[in] iVoxel Index of current voxel in the read buffer
 * @param[in] iV index of neighbouring voxel in the read buffer
 */
__device__ __forceinline__ void updateVoxel(const bool condition, const  Voxel * readData, Voxel* writeData, const int iVoxel, const int iV) {
	
	if(condition && (&readData[iV])->status == ICE) {
		const float change = transferHeat(readData, iVoxel, iV);
		if((&readData[iV])->temperature > (&readData[iVoxel])->temperature) {
			(&writeData[iVoxel])->temperature += change;
		}
		else {
			(&writeData[iVoxel])->temperature -= change;
		}
	}
	else {
		(&writeData[iVoxel])->temperature += ambientHeat(readData, iVoxel);
	}
}

/**
 * Kernel, that updates the grid
 * 
 * @param[in] readData Pointer to device read buffer
 * @param[in,out] writeData Pointer to device write buffer
 * @param[out] icedata Number of voxels that have melted
 */
__global__ void updateParticlesKernel(const Voxel * readData, Voxel * writeData, int * icedata) {
	const unsigned long blockId = blockIdx.x
								+ blockIdx.y * gridDim.x
								+ blockIdx.z * gridDim.x * gridDim.y;
	const unsigned long threadInBlock = threadIdx.x 
									  + threadIdx.y * blockDim.x 
									  + threadIdx.z * blockDim.x * blockDim.y;
	
	const unsigned long threadId = threadInBlock + blockId * blockDim.x * blockDim.y * blockDim.z;
	
	cache[threadInBlock] = 0;//clear cache
	
	const Voxel * readVoxel;
	Voxel * writeVoxel;

	if(threadId < DATA_SIZE) { //check if threadId is withing data range
		readVoxel = &readData[threadId];
		writeVoxel = &writeData[threadId];
		
		if(readVoxel->status == ICE) {
			int k = threadId / (WIDTH_HEIGHT);
			int j = (threadId - (k*WIDTH_HEIGHT))/WIDTH;
			int i = threadId - j*WIDTH - k*WIDTH_HEIGHT;
			
			//find out neighbouting particles
			updateVoxel(i+1 < WIDTH, readData, writeData, threadId, DATA_INDEX(i+1,j,k));
			updateVoxel(j+1 < HEIGHT, readData, writeData, threadId, DATA_INDEX(i,j+1,k));
			updateVoxel(k+1 < DEPTH, readData, writeData, threadId, DATA_INDEX(i,j,k+1));

			updateVoxel(i-1 >= 0, readData, writeData, threadId, DATA_INDEX(i-1,j,k));
			updateVoxel(j-1 >= 0, readData, writeData, threadId, DATA_INDEX(i,j-1,k));
			updateVoxel(k-1 >= 0, readData, writeData, threadId, DATA_INDEX(i,j,k-1));
	
			if(writeVoxel->temperature > ZERO_DEG) {
				writeVoxel->status = WATER;
				cache[threadInBlock] = 1; //this one voxel have melted
			}
		}
	}
	
	//we use parallel reduction to sum how many voxels have melted
	__syncthreads();

	int step = (BLOCK_THREADS >> 1);
	while(step > 0) {
		if (threadInBlock < step) {
			cache[threadInBlock] += cache[threadInBlock + step];
		}
		__syncthreads();
		step = (step >> 1); //only half of the threads in next iteration
	}

	if(threadId == 0) {
		*icedata = 0; //initialize the sum to 0 (in GPU memory)
	}
	__syncthreads();

	if (threadInBlock == 0) {
		atomicAdd(icedata, cache[0]); //we use atomic operation to get the total sum 
	}	
}

/**
 * This kernel initializes the data in the beginning
 * 
 * @param[in] data Pointer to device write buffer
 * @param[out] icedata Number of voxels in ICE state
 */
__global__ void initDataKernel(Voxel * data, int * icedata) {
	const unsigned long blockId = blockIdx.x
								+ blockIdx.y * gridDim.x
								+ blockIdx.z * gridDim.x * gridDim.y;
	const unsigned long threadInBlock = threadIdx.x 
									  + threadIdx.y * blockDim.x 
									  + threadIdx.z * blockDim.x * blockDim.y;
	
	const unsigned long threadId = threadInBlock + blockId * blockDim.x * blockDim.y * blockDim.z;
	
	cache[threadInBlock] = 0; //this needs to be initialized, threadId might be out of range

	if(threadId < DATA_SIZE) {
		int k = threadId / (WIDTH*HEIGHT);
		int j = (threadId - (k*WIDTH*HEIGHT))/WIDTH;
		int i = threadId - j*WIDTH - k*WIDTH*HEIGHT;

		cache[threadInBlock] = 1; //this is ice
			
		Voxel* v = &data[threadId];
		v->position[0] = i - positionOffset[0];
		v->position[1] = j - positionOffset[1];
		v->position[2] = k - positionOffset[2];

		//nechapu, proc to bez tohodle nasetovani nefunguje jak ma
		v->status = ICE;
		v->temperature = PARTICLE_INIT_TEMPERATURE;
		v->mass = PARTICLE_MASS;

		bool cond = false;
#ifdef	DATA1
		cond = (i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS);
#endif
#ifdef	DATA2
		cond = (i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS 
				|| ((i > 2*WIDTH/4 && i < 3*WIDTH/4) && (j < 2*HEIGHT/3)));
#endif
#ifdef	DATA3
		cond = (i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS 
				|| ((j < 10 || j > 4*DEPTH/5) && (i % 20 > 10)));
#endif
		if(cond) {
			v->status = AIR; //set voxel to air
			v->temperature = AIR_TEMPERATURE;
			cache[threadInBlock] = 0; //this is not ice
		}
	}
	
	//we use parallel reduction to sum how many voxels are ice
	__syncthreads();

	int step = (BLOCK_THREADS >> 1);
	while(step > 0) {
		if (threadInBlock < step) {
			cache[threadInBlock] += cache[threadInBlock + step];
		}
		__syncthreads();
		step = (step >> 1);
	}

	if(threadId == 0) {
		*icedata = 0;
	}
	__syncthreads();
	if (threadInBlock == 0) {
		atomicAdd(icedata, cache[0]);
	}	

}


void cudaInit(Voxel * readData, Voxel * writeData, int * host_ice) {
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
	
	host_write_data = writeData;
	host_read_data = readData;

	//alocate device data
	CHECK_ERR(cudaMalloc((void**)&device_write_data, DATA_SIZE * sizeof(Voxel)));
	CHECK_ERR(cudaMalloc((void**)&device_read_data, DATA_SIZE * sizeof(Voxel)));
	
	//alokace pole na GPU pro vysledky z paralelni redukce
	CHECK_ERR(cudaMalloc((void**)&device_ice_data, sizeof(int)));

	//inicializace dat
	initDataKernel<<< gridRes, blockRes >>>(device_read_data, device_ice_data);
	
	CHECK_LAST_ERR();
	//CHECK_ERR(cudaMemcpy(device_read_data, host_write_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(device_write_data, device_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToDevice));
	
	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_write_data, device_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));
	CHECK_ERR(cudaMemcpy(host_ice, device_ice_data, sizeof(int), cudaMemcpyDeviceToHost));
	
}


void cudaUpdateParticles(int * host_ice) {
	
	std::swap(device_read_data, device_write_data);

	CHECK_ERR(cudaMemcpy(device_write_data, device_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToDevice));
	
	updateParticlesKernel<<< gridRes, blockRes >>>(device_read_data, device_write_data, device_ice_data);
		
	//CHECK_LAST_ERR();

	//copy data back to CPU
	CHECK_ERR(cudaMemcpy(host_write_data, device_write_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));

	//copy the sum of voxels that have melted to CPU
	CHECK_ERR(cudaMemcpy(host_ice, device_ice_data, sizeof(int), cudaMemcpyDeviceToHost));

	//wait until GPU is done - cudaMemcpy should be blocking operation, but one never knows...
	CHECK_ERR(cudaDeviceSynchronize());
}

Voxel * cudaGetDeviceDataPointer() {
	return device_write_data;
}


void cudaFinalize() {
	CHECK_ERR(cudaFree(&device_write_data));
	CHECK_ERR(cudaFree(&device_read_data));
	CHECK_ERR(cudaFree(&device_ice_data));
}