#include "GPUSimulation.h"

GPUSimulation::GPUSimulation(int dataWidth, int dataHeight, int dataDepth) {
	this->dataCount = dataWidth * dataHeight * dataDepth;
	this->data = new Voxel[dataCount];

	this->cpumc = new CPUMarchingCubes(dataWidth, dataHeight, dataDepth);

	this->dataWidth = dataWidth;
	this->dataHeight = dataHeight;
	this->dataDepth = dataDepth;
}

GPUSimulation::~GPUSimulation() {
	delete [] this->data;
}

Voxel* GPUSimulation::getData() {
	return this->data;
}

void GPUSimulation::march() {
	this->cpumc->vMarchingCubes(this->data, this->dataCount);
}

void GPUSimulation::updateParticles() {

}
void GPUSimulation::init() {

}

void GPUSimulation::updateVoxel(bool condition, Voxel* voxel, Voxel* v) {

}


float GPUSimulation::ambientHeat(Voxel *voxel) {
	return -1;
}

float GPUSimulation::transferHeat(Voxel * voxel, Voxel* v) {
	return -1;
}