#include "GPUSimulation.h"
#include "GPUComputation.cuh"

GPUSimulation::GPUSimulation(int dataWidth, int dataHeight, int dataDepth) {
	this->dataCount = dataWidth * dataHeight * dataDepth;
	this->data = new Voxel[dataCount];

	this->cpumc = new CPUMarchingCubes();

	this->dataWidth = dataWidth;
	this->dataHeight = dataHeight;
	this->dataDepth = dataDepth;
}

GPUSimulation::~GPUSimulation() {
	cudaFinalize();
	delete [] this->data;
}

Voxel* GPUSimulation::getData() {
	return this->data;
}

//debug
void GPUSimulation::setData(Voxel * data) {
	this->data = data;
}

void GPUSimulation::march() {
	this->cpumc->vMarchingCubes(this->data, this->dataCount);
}

void GPUSimulation::updateParticles() {
	cudaUpdateParticles();
}

//data inicializuju na CPU, na GPU to jednoduše nejde :/
void GPUSimulation::init() {

	float ofsi = WIDTH/2.0f - 0.5f;
	float ofsj = HEIGHT/2.0f - 0.5f; 
	float ofsk = DEPTH/2.0f - 0.5f; 

	for(int i = 0; i < WIDTH; i++) {
		for(int j = 0; j < HEIGHT; j++) {
			for(int k = 0; k < DEPTH; k++) {
				data[DATA_INDEX(i,j,k)].setPosition(i - ofsi, j - ofsj, k - ofsk);
				if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS)
					data[DATA_INDEX(i,j,k)].status = AIR; //nastavim maly okoli na vzduch
			}
		}
	}

	cudaInit(this->data);
}

