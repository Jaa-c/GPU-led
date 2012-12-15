#include "CPUSimulation.h"

CPUSimulation::CPUSimulation(int dataWidth, int dataHeight, int dataDepth) {
	this->dataCount = dataWidth * dataHeight * dataDepth;
	this->data = new Voxel[dataCount];

	this->cpumc = new CPUMarchingCubes();

	this->dataWidth = dataWidth;
	this->dataHeight = dataHeight;
	this->dataDepth = dataDepth;
}

CPUSimulation::~CPUSimulation() {
	delete [] this->data;
}

Voxel* CPUSimulation::getData() {
	return this->data;
}

//debug
void CPUSimulation::setData(Voxel * data) {
	this->data = data;
}

void CPUSimulation::march() {
	this->cpumc->vMarchingCubes(this->data, this->dataCount);
}

void CPUSimulation::updateParticles() {

	Voxel* voxel;

	for(int i = 0; i < dataWidth; i++) {
		for(int j = 0; j < dataHeight; j++) {
			for(int k = 0; k < dataDepth; k++) {
				voxel = &data[DATA_INDEX(i,j,k)];
				if(voxel->status != ICE)
					continue;

				//okolni castice zjistim podle indexu 
				updateVoxel(i+neighbours[0][0] < dataWidth, voxel, &data[DATA_INDEX(i+neighbours[0][0],j,k)]);
				updateVoxel(j+neighbours[1][1] < dataHeight, voxel, &data[DATA_INDEX(i,j+neighbours[1][1],k)]);
				updateVoxel(k+neighbours[2][2] < dataDepth, voxel, &data[DATA_INDEX(i,j,k+neighbours[2][2])]);
				
				updateVoxel(i+neighbours[3][0] >= 0, voxel, &data[DATA_INDEX(i+neighbours[3][0],j,k)]);
				updateVoxel(j+neighbours[4][1] >= 0, voxel, &data[DATA_INDEX(i,j+neighbours[4][1],k)]);
				updateVoxel(k+neighbours[5][2] >= 0, voxel, &data[DATA_INDEX(i,j,k+neighbours[5][2])]);

				if(voxel->temperature > ZERO_DEG) {
					voxel->status = WATER;
				}

			}
		}
	}

}

void CPUSimulation::init() {

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

}

/**
 * Updatuje buòku
 *
 * @param condition - jestli je bunka uvnitr mrizky
 * @param voxel - aktualni bunka
 * @param v - sousedni bunka
 */
void CPUSimulation::updateVoxel(bool condition, Voxel* voxel, Voxel* v) {
	if(condition) {
		if(v->status != ICE)
			voxel->temperature += ambientHeat(voxel);
		else {
			float change = transferHeat(voxel, v);
			v->temperature += change;
			voxel->temperature -= change;
		}
	}
	else {
		voxel->temperature += ambientHeat(voxel);
	}
}

/**
 * Dodá buòce "ambientní" teplotu okolí - tedy teplotu od vzduchu
 *
 * @param voxel - aktualni buòka
 */
float CPUSimulation::ambientHeat(Voxel *voxel) {
	return TIME_STEP * (
		(THERMAL_CONDUCTIVITY * (AIR_TEMPERATURE - voxel->temperature))
		/ (SPECIFIC_HEAT_CAP_ICE * voxel->mass)
		);
}

/**
 * Vypoète kolik tepla se pøenese mezi èásticemi
 *
 * @param voxel - aktualni buòka
 * @param v - sousední buòka
 */
float CPUSimulation::transferHeat(Voxel * voxel, Voxel* v) {
	//TODO:: zapocitat do vzorecku hustotu materialu
	if(voxel->status == ICE)
		return TIME_STEP * (THERMAL_DIFFUSION_ICE * v->mass * (v->temperature - voxel->temperature) / DENSITY_ICE);
	else if(voxel->status == WATER)
		return TIME_STEP * (THERMAL_DIFFUSION_WATER * v->mass * (v->temperature - voxel->temperature) / DENSITY_WATER);
	else
		return 0;
}