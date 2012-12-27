#include "cudaCommon.cuh"

/************************************************
*			   Ukazatele na data			    *
************************************************/
Voxel * device_write_data = 0;
Voxel * device_read_data = 0;
Voxel * host_write_data = 0;
Voxel * host_read_data = 0;

int * device_ice_data = 0;

/************************************************
*		   Velikost mrizky a bloku			    *
************************************************/
#define	BLOCK_THREADS	512
#define	GRID_RES		((int) (pow(DATA_SIZE / BLOCK_THREADS, 1.0f / 3.0f) + 1.0f))
#define GRID_SIZE		(GRID_RES*GRID_RES*GRID_RES)

const dim3 blockRes(8,8,8);//512 vlaken
const dim3 gridRes(GRID_RES, GRID_RES, GRID_RES); //v zavislosti na datech

/************************************************
*				Deklarace pameti			    *
************************************************/
__shared__ int cache[BLOCK_THREADS];
__constant__ float  positionOffset[3] = { WIDTH/2.0f - 0.5f, HEIGHT/2.0f - 0.5f, DEPTH/2.0f - 0.5f};
__constant__ float thermal_diffusion[3] = {THERMAL_DIFFUSION_ICE, THERMAL_DIFFUSION_WATER, 1.0f};
__constant__ float density[3] = {DENSITY_ICE, DENSITY_WATER, 1.0f};
__constant__ int transfer[3] = {1, 1, 0};


/************************************************
*				Device metody				    *
************************************************/
__device__ __forceinline__ float ambientHeat(const Voxel * data, const int ivoxel) {
	return TIME_STEP * (
		(THERMAL_CONDUCTIVITY * (AIR_TEMPERATURE - (&data[ivoxel])->temperature))
		/ (SPECIFIC_HEAT_CAP_ICE * (&data[ivoxel])->mass)
		);
}

__device__ __forceinline__ float transferHeat(const Voxel * data, const int ivoxel, const int iv) {

	return transfer[(&data[ivoxel])->status] * 
		(TIME_STEP * (thermal_diffusion[(&data[ivoxel])->status] * 
		(&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / density[(&data[ivoxel])->status]));
	
	//const Voxel * voxel = (&data[ivoxel]);
	//const Voxel * v = &data[iv];
	//if((&data[ivoxel])->status == ICE)
	//	return TIME_STEP * (THERMAL_DIFFUSION_ICE * (&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / DENSITY_ICE);
	//else if((&data[ivoxel])->status == WATER)
	//	return TIME_STEP * (THERMAL_DIFFUSION_WATER * (&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / DENSITY_WATER);
	//else
	//	return 0;
}

__shared__ float tempChange[BLOCK_THREADS];
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

/************************************************
/ Kernel, ktery upravuje castice v kazde iteraci
************************************************/
__global__ void updateParticlesKernel(const Voxel * readData, Voxel * writeData, int * icedata) {
	const unsigned long blockId = blockIdx.x
								+ blockIdx.y * gridDim.x
								+ blockIdx.z * gridDim.x * gridDim.y;
	const unsigned long threadInBlock = threadIdx.x 
									  + threadIdx.y * blockDim.x 
									  + threadIdx.z * blockDim.x * blockDim.y;
	
	const unsigned long threadId = threadInBlock + blockId * blockDim.x * blockDim.y * blockDim.z;
	
	cache[threadInBlock] = 0;//vynulejeme cache
	
	const Voxel * readVoxel;
	Voxel * writeVoxel;

	if(threadId < DATA_SIZE) { //pokud neni index vlakna mimo data
		readVoxel = &readData[threadId];
		writeVoxel = &writeData[threadId];
		
		if(readVoxel->status == ICE) {
			int k = threadId / (WIDTH_HEIGHT);
			int j = (threadId - (k*WIDTH_HEIGHT))/WIDTH;
			int i = threadId - j*WIDTH - k*WIDTH_HEIGHT;
			
			//okolni castice zjistim podle indexu
			updateVoxel(i+1 < WIDTH, readData, writeData, threadId, DATA_INDEX(i+1,j,k));
			updateVoxel(j+1 < HEIGHT, readData, writeData, threadId, DATA_INDEX(i,j+1,k));
			updateVoxel(k+1 < DEPTH, readData, writeData, threadId, DATA_INDEX(i,j,k+1));

			updateVoxel(i-1 >= 0, readData, writeData, threadId, DATA_INDEX(i-1,j,k));
			updateVoxel(j-1 >= 0, readData, writeData, threadId, DATA_INDEX(i,j-1,k));
			updateVoxel(k-1 >= 0, readData, writeData, threadId, DATA_INDEX(i,j,k-1));
	
			//__syncthreads();
			if(writeVoxel->temperature > ZERO_DEG) {
				writeVoxel->status = WATER;
				cache[threadInBlock] = 1; //kolik bunek ledu roztalo?
			}
		}
	}
	
	//redukce pro vsechny vlakna
	__syncthreads(); // synchronizace všech vláken

	int step = (BLOCK_THREADS >> 1);
	while(step > 0) {
		if (threadInBlock < step) {
			cache[threadInBlock] += cache[threadInBlock + step];
		}
		__syncthreads(); // synchronizace vláken po provedení každé fáze
		step = (step >> 1); // zmenšení kroku pro další fázi redukce
	}

	if(threadId == 0) {
		*icedata = 0;//inicializace na 0
	}
	__syncthreads();
	if (threadInBlock == 0) {
		atomicAdd(icedata, cache[0]);
	}	
}


__global__ void initDataKernel(Voxel * data, int * icedata) {
	const unsigned long blockId = blockIdx.x
								+ blockIdx.y * gridDim.x
								+ blockIdx.z * gridDim.x * gridDim.y;
	const unsigned long threadInBlock = threadIdx.x 
									  + threadIdx.y * blockDim.x 
									  + threadIdx.z * blockDim.x * blockDim.y;
	
	const unsigned long threadId = threadInBlock + blockId * blockDim.x * blockDim.y * blockDim.z;
	cache[threadInBlock] = 0;

	if(threadId < DATA_SIZE) {
		int k = threadId / (WIDTH*HEIGHT);
		int j = (threadId - (k*WIDTH*HEIGHT))/WIDTH;
		int i = threadId - j*WIDTH - k*WIDTH*HEIGHT;

		cache[threadInBlock] = 1;
			
		Voxel* v = &data[threadId];
		v->position[0] = i - positionOffset[0];
		v->position[1] = j - positionOffset[1];
		v->position[2] = k - positionOffset[2];
		v->status = ICE;
		v->temperature = PARTICLE_INIT_TEMPERATURE;

#ifdef DATA1
		if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS) {
			v->status = AIR; //nastavim maly okoli na vzduch
			v->temperature = AIR_TEMPERATURE;
			cache[threadInBlock] = 0; //kolik bunek ledu roztalo?
		}
	}
#endif
#ifdef DATA2
		if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS 
			|| ((i > 2*WIDTH/4 && i < 3*WIDTH/4) && (j < 2*HEIGHT/3))) {
			v->status = AIR; //nastavim maly okoli na vzduch
			v->temperature = AIR_TEMPERATURE;
			cache[threadInBlock] = 0; //kolik bunek ledu roztalo?
		}
	}
#endif
#ifdef DATA3
		if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS 
			|| ((i < 10) && (j % 20 > 10))) {
			v->status = AIR; //nastavim maly okoli na vzduch
			v->temperature = AIR_TEMPERATURE;
			cache[threadInBlock] = 0; //kolik bunek ledu roztalo?
		}
	}
#endif
	//redukce pro vsechny vlakna
	__syncthreads(); // synchronizace všech vláken

	int step = (BLOCK_THREADS >> 1);
	while(step > 0) {
		if (threadInBlock < step) {
			cache[threadInBlock] += cache[threadInBlock + step];
		}
		__syncthreads(); // synchronizace vláken po provedení každé fáze
		step = (step >> 1); // zmenšení kroku pro další fázi redukce
	}

	if(threadId == 0) {
		*icedata = 0;//inicializace na 0 v paměti gpu
	}
	__syncthreads();
	if (threadInBlock == 0) {
		atomicAdd(icedata, cache[0]);
	}	

}


/************************************************
/ Inicializace cudy
************************************************/
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
	CHECK_ERR(cudaMemcpy(device_write_data, device_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToDevice));
	
	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_write_data, device_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));
	CHECK_ERR(cudaMemcpy(host_ice, device_ice_data, sizeof(int), cudaMemcpyDeviceToHost));
	
}


/************************************************
/ metoda, ktera vola kernel pro update mrizky
************************************************/
void cudaUpdateParticles(int * host_ice) {
	
	std::swap(device_read_data, device_write_data);

	CHECK_ERR(cudaMemcpy(device_write_data, device_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToDevice));
	
	updateParticlesKernel<<< gridRes, blockRes >>>(device_read_data, device_write_data, device_ice_data);
		
	//CHECK_LAST_ERR();

	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_write_data, device_write_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));

	//prekopirujeme pocet roztatych bunek na CPU
	CHECK_ERR(cudaMemcpy(host_ice, device_ice_data, sizeof(int), cudaMemcpyDeviceToHost));

	//pockame nez se dokonci kopirovani dat - asi neni treba, cudaMemcpy je pry blokujici operace
	CHECK_ERR(cudaDeviceSynchronize());
}

Voxel * cudaGetDeviceDataPointer() {
	return device_write_data;
}

//uvolni prostredky alokovane cudou
void cudaFinalize() {
	CHECK_ERR(cudaFree(&device_write_data));
	CHECK_ERR(cudaFree(&device_read_data));
	CHECK_ERR(cudaFree(&device_ice_data));
}