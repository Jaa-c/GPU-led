#ifndef __GPUCOMPUTATION_CUH__
#define __GPUCOMPUTATION_CUH__

//inicializace cudy
void cudaInit(Voxel * host_data);

//uklizení po cudì
void cudaFinalize();

//aktualizace mrizky
void cudaUpdateParticles();

//vraci ukazatel na data v pameti GPU
Voxel * cudaGetHostDataPointer();

//inicializace marching cubes, volá se jen jednou
void cudaMarchInit();

//marching cubes v cudì
void cudaMarchingCubes();

#endif