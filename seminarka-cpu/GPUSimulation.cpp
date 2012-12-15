#include "GPUSimulation.h"
#include "GPUComputation.cuh"

GPUSimulation::GPUSimulation(int dataWidth, int dataHeight, int dataDepth) {
	this->dataCount = dataWidth * dataHeight * dataDepth;
	this->data = new Voxel[dataCount];

	this->cpumc = new CPUMarchingCubes(dataWidth, dataHeight, dataDepth);

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

	float ofsi = dataWidth/2.0f - 0.5f;
	float ofsj = dataHeight/2.0f - 0.5f; 
	float ofsk = dataDepth/2.0f - 0.5f; 

	int ii, ij, ik, index = 0;
	ii = 0;
	for(float i = -ofsi; i <= ofsi; i++) {
		ij = 0;
		for(float j = -ofsj; j <= ofsj; j++) {
			ik = 0;
			for(float k = -ofsk; k <= ofsk; k++) {

				data[DATA_INDEX(ii,ij,ik)].setPosition(i, j, k);

				if(ii <= 1 || ij <= 1 || ik <= 1)
					data[DATA_INDEX(ii,ij,ik)].status = AIR; //nastavim maly okoli na vzduch

				ik++;
			}
			ij++;
		}
		ii++;
	}

	cudaInit(this->data);
}

