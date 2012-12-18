#ifndef __GPUCOMPUTATION_CUH__
#define __GPUCOMPUTATION_CUH__

//inicializace cudy
void cudaInit(Voxel * readData, Voxel * writeData, int * ice);

//uklizení po cudì
void cudaFinalize();

//aktualizace mrizky
void cudaUpdateParticles(int * ice);

//vraci ukazatel na data v pameti GPU
Voxel * cudaGetDeviceDataPointer();

//inicializace marching cubes, volá se jen jednou
void cudaMarchInit(Voxel * host_data);

//marching cubes v cudì
void cudaMarchingCubes();

#endif