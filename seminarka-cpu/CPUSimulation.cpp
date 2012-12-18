#include "CPUSimulation.h"

CPUSimulation::CPUSimulation() {
	this->writeData = new Voxel[DATA_SIZE];
	this->readData = new Voxel[DATA_SIZE];

	this->cpumc = new CPUMarchingCubes();
}

CPUSimulation::~CPUSimulation() {
	delete [] this->writeData;
	delete [] this->readData;
}

Voxel* CPUSimulation::getData() {
	return NULL;// this->writeData;
}

//debug
void CPUSimulation::setData(Voxel * readData) {
	//this->readData = readData;
}

void CPUSimulation::march() {
	this->cpumc->vMarchingCubes(this->writeData);
}

int CPUSimulation::updateParticles() {

	std::swap(writeData, readData);

	Voxel * readVoxel, * writeVoxel;

	for(int i = 0; i < WIDTH; i++) {
		for(int j = 0; j < HEIGHT; j++) {
			for(int k = 0; k < DEPTH; k++) {
				writeVoxel = &writeData[DATA_INDEX(i,j,k)];
				readVoxel = &readData[DATA_INDEX(i,j,k)];

				*writeVoxel = *readVoxel; //nastavime aktualni stav

				if(readVoxel->status != ICE) {
					continue;
				}

				//okolni castice zjistim podle indexu 
				updateVoxel(i+1 < WIDTH, writeVoxel, &writeData[DATA_INDEX(i+1,j,k)], readVoxel, &readData[DATA_INDEX(i+1,j,k)]);
				updateVoxel(j+1 < HEIGHT, writeVoxel, &writeData[DATA_INDEX(i,j+1,k)], readVoxel, &readData[DATA_INDEX(i,j+1,k)]);
				updateVoxel(k+1 < DEPTH, writeVoxel, &writeData[DATA_INDEX(i,j,k+1)], readVoxel, &readData[DATA_INDEX(i,j,k+1)]);
				
				updateVoxel(i-1 >= 0, writeVoxel, &writeData[DATA_INDEX(i-1,j,k)], readVoxel, &readData[DATA_INDEX(i-1,j,k)]);
				updateVoxel(j-1 >= 0, writeVoxel, &writeData[DATA_INDEX(i,j-1,k)] ,readVoxel, &readData[DATA_INDEX(i,j-1,k)]);
				updateVoxel(k-1 >= 0, writeVoxel, &writeData[DATA_INDEX(i,j,k-1)], readVoxel, &readData[DATA_INDEX(i,j,k-1)]);

				if(writeVoxel->temperature > ZERO_DEG) {
					writeVoxel->status = WATER;
					this->iceParticles--;
				}

			}
		}
	}

	return this->iceParticles; 

}

void CPUSimulation::init() {

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


}

/**
 * Updatuje buòku
 *
 * @param condition - jestli je bunka uvnitr mrizky
 * @param voxel - aktualni bunka
 * @param v - sousedni bunka
 */
void CPUSimulation::updateVoxel(bool condition, Voxel * writeVoxel,  Voxel * writeV , Voxel* readVoxel, Voxel* readV) {
	if(condition && readV->status == ICE) {
		float change = transferHeat(readVoxel, readV);
		writeV->temperature += change;
		writeVoxel->temperature -= change;
	}
	else {
		writeVoxel->temperature += ambientHeat(readVoxel);
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