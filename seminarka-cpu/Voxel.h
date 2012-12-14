#ifndef __VOXEL_H__
#define __VOXEL_H__

#include "defines.h"
#include "structures.h"

enum Status {
	ICE = 0,
	WATER,
	AIR
};

struct Voxel {
	float position[3];
	float temperature;
	float velocity;
	Status status;
	float radius;//?
	float area;//?
	float mass;

	Voxel(float x, float y, float z);

	Voxel();

	void setPosition(float x, float y, float z);
};

#endif