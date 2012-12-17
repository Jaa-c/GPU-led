#include "CPUSimulation.h"

CPUSimulation::CPUSimulation() {
	this->data = new Voxel[DATA_SIZE];

	this->cpumc = new CPUMarchingCubes();
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
	this->cpumc->vMarchingCubes(this->data);
}

int CPUSimulation::updateParticles() {

	Voxel* voxel;

	for(int i = 0; i < WIDTH; i++) {
		for(int j = 0; j < HEIGHT; j++) {
			for(int k = 0; k < DEPTH; k++) {
				voxel = &data[DATA_INDEX(i,j,k)];
				if(voxel->status != ICE)
					continue;

				//okolni castice zjistim podle indexu 
				updateVoxel(i+1 < WIDTH, voxel, &data[DATA_INDEX(i+1,j,k)]);
				updateVoxel(j+1 < HEIGHT, voxel, &data[DATA_INDEX(i,j+1,k)]);
				updateVoxel(k+1 < DEPTH, voxel, &data[DATA_INDEX(i,j,k+1)]);
				
				updateVoxel(i-1 >= 0, voxel, &data[DATA_INDEX(i-1,j,k)]);
				updateVoxel(j-1 >= 0, voxel, &data[DATA_INDEX(i,j-1,k)]);
				updateVoxel(k-1 >= 0, voxel, &data[DATA_INDEX(i,j,k-1)]);

				if(voxel->temperature > ZERO_DEG) {
					voxel->status = WATER;
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
				data[DATA_INDEX(i,j,k)].setPosition(i - ofsi, j - ofsj, k - ofsk);
				if(i < AIR_VOXELS || j < AIR_VOXELS || k < AIR_VOXELS) {
					data[DATA_INDEX(i,j,k)].status = AIR; //nastavim maly okoli na vzduch
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