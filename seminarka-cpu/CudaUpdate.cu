#include "cudaCommon.cuh"

/************************************************
*			   Ukazatele na data			    *
************************************************/
Voxel * device_write_data = 0;
Voxel * device_read_data = 0;
Voxel * host_write_data = 0;
Voxel * host_read_data = 0;

int * ice_data = 0;
int * host_ice_data = 0;

/************************************************
*		   Velikost mrizky a bloku			    *
************************************************/
//128 - 6461
//512 - 6607
//1024 - 8500

#define	BLOCK_THREADS	128
#define	GRID_RES		((int) (pow(DATA_SIZE / BLOCK_THREADS, 1.0 / 3.0) + 0.5f) +1)
#define GRID_SIZE		(GRID_RES*GRID_RES*GRID_RES)

const dim3 blockRes(8,4,4);//1024 vlaken
const dim3 gridRes(GRID_RES, GRID_RES, GRID_RES); //v zavislosti na datech

/************************************************
*				Deklarace pameti			    *
************************************************/
__shared__ int cache[BLOCK_THREADS];


//__device__ unsigned long getThreadId() {
//	const unsigned long block = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
//	
//	return  block * blockDim.x * blockDim.y * blockDim.z +
//			threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//}


/************************************************
*				Device metody				    *
************************************************/
__device__ float ambientHeat(const Voxel * data, const int ivoxel) {
	//const Voxel * voxel = (&data[ivoxel]);
	return TIME_STEP * (
		(THERMAL_CONDUCTIVITY * (AIR_TEMPERATURE - (&data[ivoxel])->temperature))
		/ (SPECIFIC_HEAT_CAP_ICE * (&data[ivoxel])->mass)
		);
}

__constant__ float thermal_diffusion[3] = {THERMAL_DIFFUSION_ICE, THERMAL_DIFFUSION_WATER, 1.0f};
__constant__ float density[3] = {DENSITY_ICE, DENSITY_WATER, 1.0f};
__constant__ int transfer[3] = {1, 1, 0};

__device__ float transferHeat(const Voxel * data, const int ivoxel, const int iv) {
	//const Voxel * voxel = (&data[ivoxel]);
	//const Voxel * v = &data[iv];
	//return transfer[(&data[ivoxel])->status] * (TIME_STEP * (thermal_diffusion[(&data[ivoxel])->status] * (&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / density[(&data[ivoxel])->status]));
	if((&data[ivoxel])->status == ICE)
		return TIME_STEP * (THERMAL_DIFFUSION_ICE * (&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / DENSITY_ICE);
	else if((&data[ivoxel])->status == WATER)
		return TIME_STEP * (THERMAL_DIFFUSION_WATER * (&data[iv])->mass * ((&data[iv])->temperature - (&data[ivoxel])->temperature) / DENSITY_WATER);
	else
		return 0;
}


__device__ void updateVoxel(const bool condition, const  Voxel * readData, Voxel* writeData, const int iVoxel, const int iV) {
	//Voxel * writeVoxel = (&writeData[iVoxel]);
	//Voxel * writeV = (&writeData[iV]);
	//const Voxel * readV = &readData[iV];

	if(condition && (&readData[iV])->status == ICE) {
		const float change = transferHeat(readData, iVoxel, iV);
		(&writeData[iV])->temperature += change;
		(&writeData[iVoxel])->temperature -= change;
	}
	else {
		(&writeData[iVoxel])->temperature += ambientHeat(readData, iVoxel);
	}
}


/************************************************
/ Kernel, ktery upravuje castice v kazde iteraci
************************************************/
__global__ void updateParticlesKernel(const Voxel * readData, Voxel * writeData, int * icedata) {

	//const unsigned long threadId = getThreadId();
	const unsigned long blockId = blockIdx.x
								+ blockIdx.y * gridDim.x
								+ blockIdx.z * gridDim.x * gridDim.y;
	const unsigned long threadInBlock = threadIdx.x 
									  + threadIdx.y * blockDim.x 
									  + threadIdx.z * blockDim.x * blockDim.y;
	
	int threadId = threadInBlock + blockId * blockDim.x * blockDim.y * blockDim.z;
	
	cache[threadInBlock] = 0;//vynulejeme cache
	
	const Voxel * readVoxel;
	Voxel * writeVoxel;

	if(threadId < DATA_SIZE) { //pokud neni index vlakna mimo data
		readVoxel = &readData[threadId];
		writeVoxel = &writeData[threadId];

		*writeVoxel = *readVoxel; //nastavime aktualni stav

		if(readVoxel->status == ICE) {

			int k = threadId / (WIDTH*HEIGHT);
			int j = (threadId - (k*WIDTH*HEIGHT))/WIDTH;
			int i = threadId - j*WIDTH - k*WIDTH*HEIGHT;

			//okolni castice zjistim podle indexu 
			updateVoxel(i+1 < WIDTH, readData, writeData, threadId, DATA_INDEX(i+1,j,k));
			updateVoxel(j+1 < HEIGHT, readData, writeData, threadId, DATA_INDEX(i,j+1,k));
			updateVoxel(k+1 < DEPTH, readData, writeData, threadId, DATA_INDEX(i,j,k+1));
				
			updateVoxel(i-1 >= 0, readData, writeData, threadId, DATA_INDEX(i-1,j,k));
			updateVoxel(j-1 >= 0, readData, writeData, threadId, DATA_INDEX(i,j-1,k));
			updateVoxel(k-1 >= 0, readData, writeData, threadId, DATA_INDEX(i,j,k-1));
	
			
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

	if (threadInBlock == 0) {
		icedata[blockId] = cache[0];
	}
	__syncthreads(); // mozna zbytecny?
	
	//je potreba zarovnat GRID_SIZE na mocninu dvou
	if(false && threadId <= GRID_SIZE) {
		step = (GRID_SIZE >> 1);
		while(step > 1) {
			if (threadId < step) {
				icedata[threadId] += icedata[threadId + step];
			}
			__syncthreads(); // synchronizace vláken po provedení každé fáze
			step = (step >> 1); // zmenšení kroku pro další fázi redukce
		}
		//v icedata[0] by mel bejt spravnej vysledek
	}
	
}

//__global__ void initDataKernel(Voxel * data) {
//	const unsigned long threadId = getThreadId();
//
//	if(threadId > DATA_SIZE)
//		return;
//
//	int k = threadId / (WIDTH*HEIGHT);
//	int j = (threadId - (k*WIDTH*HEIGHT))/WIDTH;
//	int i = threadId - j*WIDTH - k*WIDTH*HEIGHT;
//
//	//shared memory
//	float ofsi = 0;//WIDTH/2.0f - 0.5f;
//	float ofsj = 0;//HEIGHT/2.0f - 0.5f;
//	float ofsk = 0;//DEPTH/2.0f - 0.5f;
//	
//	Voxel* v = &data[threadId];
//	v->position[0] = i - ofsi;
//	v->position[1] = j - ofsj;
//	v->position[2] = k - ofsk;
//		
//	if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS)
//		v->status = AIR; //nastavim maly okoli na vzduch
//
//}


/************************************************
/ Inicializace cudy
************************************************/
void cudaInit(Voxel * readData, Voxel * writeData, int * ice) {
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
	
	//CHECK_ERR(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
	
	host_write_data = writeData;
	host_read_data = readData;

	//alocate device data
	CHECK_ERR(cudaMalloc((void**)&device_write_data, DATA_SIZE * sizeof(Voxel)));
	CHECK_ERR(cudaMalloc((void**)&device_read_data, DATA_SIZE * sizeof(Voxel)));
	
	//CHECK_ERR(cudaMalloc((void**)&ice_particles, sizeof(int)));
	
	//alokace pole na GPU pro vysledky z paralelni redukce
	CHECK_ERR(cudaMalloc((void**)&ice_data, GRID_SIZE * sizeof(int)));
	//alokace stejneho pole na CPU
	host_ice_data = (int *) malloc(GRID_SIZE * sizeof(int));

	//copy data from host to device
	CHECK_ERR(cudaMemcpy(device_read_data, host_read_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(device_write_data, host_write_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyHostToDevice));//zbytecny asi?

	//CHECK_ERR(cudaMemcpy(ice_particles, ice, sizeof(int), cudaMemcpyHostToDevice));

	/*dim3 gridRes(32,32,32);
	dim3 blockRes(8,8,8);
	initDataKernel<<< gridRes, blockRes >>>(device_data);
	CHECK_LAST_ERR();
	zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_data, device_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));*/
	
}


/************************************************
/ metoda, ktera vola kernel pro update mrizky
************************************************/
void cudaUpdateParticles(int * host_ice) {
	
	std::swap(device_read_data, device_write_data);

	updateParticlesKernel<<< gridRes, blockRes >>>(device_read_data, device_write_data, ice_data);
	
	//CHECK_LAST_ERR();

	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_write_data, device_write_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));

	//prekopirujeme pocet roztatych bunek na CPU
	CHECK_ERR(cudaMemcpy(host_ice_data, ice_data, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

	//CPU zlocin
	for(int i = 0; i < GRID_SIZE; i++) {
		*host_ice += host_ice_data[i];
	}

	//pockame nez se dokonci kopirovani dat ... ??
	//CHECK_ERR(cudaDeviceSynchronize());
}

Voxel * cudaGetDeviceDataPointer() {
	return NULL;//host_write_data;
}

//uvolni prostredky alokovane cudou
void cudaFinalize() {
	CHECK_ERR(cudaFree(&device_write_data));
	CHECK_ERR(cudaFree(&device_read_data));
	CHECK_ERR(cudaFree(&ice_data));
}