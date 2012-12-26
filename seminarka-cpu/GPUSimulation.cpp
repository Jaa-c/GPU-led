
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
	return this->writeData;
}

//debug
void GPUSimulation::setData(Voxel * data) {
	//this->readData = data;
}

void GPUSimulation::march() {
	this->cpumc->vMarchingCubes(this->writeData);
}

int GPUSimulation::updateParticles() {
	int melt = 0;//kolik castic roztalo v aktualni iteraci
	cudaUpdateParticles(&melt);
	return this->iceParticles -= melt;

}

//data inicializuju na CPU, na GPU to jednoduše nejde :/
void GPUSimulation::init() {
	cudaInit(this->readData, this->writeData, &this->iceParticles);
}

