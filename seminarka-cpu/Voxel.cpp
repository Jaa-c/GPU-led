#include <iostream>
#include "voxel.h"

Voxel::Voxel(const float x, const float y, const float z) : temperature(PARTICLE_INIT_TEMPERATURE), velocity(0.0f), status(ICE), mass(PARTICLE_MASS) {
		position[0] = x;
		position[1] = y;
		position[2] = z;
	}

Voxel::Voxel() : temperature(PARTICLE_INIT_TEMPERATURE), velocity(0.0f), status(ICE), mass(PARTICLE_MASS) {
	//if(PARTICLE_RANDOM_MASS)
	//	mass += sign[rand() % 2] * (rand() % 5) * RAND_MASS_SIZE;
}

void Voxel::setPosition(const float x, const float y, const float z) {
	position[0] = x;
	position[1] = y;
	position[2] = z;
}