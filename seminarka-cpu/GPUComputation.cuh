#ifndef __GPUCOMPUTATION_CUH__
#define __GPUCOMPUTATION_CUH__

//inicializace cudy
void cudaInit(Voxel * host_data);

//uklizen� po cud�
void cudaFinalize();

//aktualizace mrizky
void cudaUpdateParticles();

//vraci ukazatel na data v pameti GPU
Voxel * cudaGetHostDataPointer();

//inicializace marching cubes, vol� se jen jednou
void cudaMarchInit();

//marching cubes v cud�
void cudaMarchingCubes();

#endif