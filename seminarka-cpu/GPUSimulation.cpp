
#include <iostream>
#include "GPUSimulation.h"
#include "GPUComputation.cuh"

GPUSimulation::GPUSimulation() {
	this->writeData = new Voxel[DATA_SIZE];
	this->readData = new Voxel[DATA_SIZE];

	this->cpumc = new CPUMarchingCubes();

}

GPUSimulation::~GPUSimulation() {
	cudaFinalize();
	delete [] this->writeData;
	delete [] this->readData;
}

//debug
Voxel* GPUSimulation::getData() {
	return NULL;// this->writeData;
}

//debug
void GPUSimulation::setData(Voxel * data) {
	//this->readData = data;
}

void GPUSimulation::march() {
	this->cpumc->vMarchingCubes(this->writeData);
}

int GPUSimulation::updateParticles() {
	
	//std::swap(writeData, readData);

	int melt = 0;//kolik castic roztalo v aktualni iteraci
	cudaUpdateParticles(&melt);
	//int c = 0;
	//for(int i = 0; i < 216; i++) {
	//	c += melt[i];
	//	//if(melt[i] != 0)
	//	//	std::cout << melt[i] << " ";
	//}
	////std::cout << "\n";
	this->iceParticles -= melt;
	return this->iceParticles;
}

//data inicializuju na CPU, na GPU to jednoduše nejde :/
void GPUSimulation::init() {

	float ofsi = WIDTH/2.0f - 0.5f;
	float ofsj = HEIGHT/2.0f - 0.5f; 
	float ofsk = DEPTH/2.0f - 0.5f; 

	this->iceParticles = DATA_SIZE;

	for(int i = 0; i < WIDTH; i++) {
		for(int j = 0; j < HEIGHT; j++) {
			for(int k = 0; k < DEPTH; k++) {
				writeData[DATA_INDEX(i,j,k)].setPosition(i - ofsi, j - ofsj, k - ofsk);
				if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS) {
					writeData[DATA_INDEX(i,j,k)].status = AIR; //nastavim maly okoli na vzduch
					this->iceParticles--;
				}
			}
		}
	}

	cudaInit(this->readData, this->writeData, &this->iceParticles);
}

