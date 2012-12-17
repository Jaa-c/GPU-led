#include "cudaCommon.cuh"

/************************************************
*				Device buffers			    *
************************************************/
Voxel * device_data = 0;
Voxel * host_data = 0;

//int * ice_particles = 0;
int * ice_data = 0;

/************************************************
*		   Grid and block resolution		    *
************************************************/
#define	BLOCK_THREADS	1024
#define	GRID_RES		((int) (pow(DATA_SIZE / BLOCK_THREADS, 1.0 / 3.0) + 0.5f))
#define GRID_SIZE		(GRID_RES*GRID_RES*GRID_RES)

const dim3 blockRes(16,8,8);//1024 vlaken
const dim3 gridRes(GRID_RES, GRID_RES, GRID_RES); //v zavislosti na datech

/************************************************
*				Memory declarations			    *
************************************************/
__shared__ int cache[BLOCK_THREADS];

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

__global__ void updateParticlesKernel(Voxel * data, int * icedata) {

	//const unsigned long threadId = getThreadId();
	const unsigned long blockId = blockIdx.x
								+ blockIdx.y * gridDim.x
								+ blockIdx.z * gridDim.x * gridDim.y;
	const unsigned long threadInBlock = threadIdx.x 
									  + threadIdx.y * blockDim.x 
									  + threadIdx.z * blockDim.x * blockDim.y;
	
	int threadId = threadInBlock + blockId * blockDim.x * blockDim.y * blockDim.z;
	
	cache[threadInBlock] = 0;//vynulejeme cache, aby neobsahovala nahodna data

	if(threadId < DATA_SIZE) { //pokud neni index vlakna mimo data
		Voxel * voxel = &data[threadId];
		if(voxel->status == ICE) {

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
				cache[threadInBlock] = 1; //kolik bunek ledu roztalo?
			}
		}
	}
	
	//redukce pro vsechny vlakna

	__syncthreads(); // synchronizace všech vláken

	int step = BLOCK_THREADS / 2;
	while(step > 0) {
		if (threadInBlock < step) {
			cache[threadInBlock] += cache[threadInBlock + step];
		}
		__syncthreads(); // synchronizace vláken po provedení každé fáze
		step /= 2; // zmenšení kroku pro další fázi redukce
	}

	if (threadInBlock == 0) {
		if(cache[0] > threadInBlock) {//debug - proè??
			(&data[99999999])->area = 0;
		}
		icedata[blockId] = cache[0];
	}
	__syncthreads(); // mozna zbytecny?
	

	if(false && threadId <= GRID_SIZE) {
		step = GRID_SIZE / 2;
		while(step > 1) {
			if (threadId < step) {
				icedata[threadId] += icedata[threadId + step];
			}
			__syncthreads(); // synchronizace vláken po provedení každé fáze
			step /= 2; // zmenšení kroku pro další fázi redukce
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


void cudaInit(Voxel * data, int * ice) {
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

	host_data = data;

	//alocate host data
	CHECK_ERR(cudaMalloc((void**)&device_data, DATA_SIZE * sizeof(Voxel)));
	
	//CHECK_ERR(cudaMalloc((void**)&ice_particles, sizeof(int)));

	CHECK_ERR(cudaMalloc((void**)&ice_data, GRID_SIZE * sizeof(int)));

	//copy data from host to device
	CHECK_ERR(cudaMemcpy(device_data, host_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyHostToDevice));

	//CHECK_ERR(cudaMemcpy(ice_particles, ice, sizeof(int), cudaMemcpyHostToDevice));

	/*dim3 gridRes(32,32,32);
	dim3 blockRes(8,8,8);
	initDataKernel<<< gridRes, blockRes >>>(device_data);
	CHECK_LAST_ERR();
	zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_data, device_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));*/
	
}


void cudaUpdateParticles(int * host_ice) {

	updateParticlesKernel<<< gridRes, blockRes >>>(device_data, ice_data);
	
	CHECK_LAST_ERR();

	//zkopirovani dat zpet na CPU
	CHECK_ERR(cudaMemcpy(host_data, device_data, DATA_SIZE * sizeof(Voxel), cudaMemcpyDeviceToHost));
	//cudaMemcpyAsync ??

	//kolik je jeste ledu
	CHECK_ERR(cudaMemcpy(host_ice, ice_data, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

	//pockame nez se dokonci kopirovani dat
	CHECK_ERR(cudaDeviceSynchronize());

}

Voxel * cudaGetHostDataPointer() {
	return host_data;
}

void cudaFinalize() {
	CHECK_ERR(cudaFree(&device_data));
}