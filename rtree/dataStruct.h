#pragma once

#define CHILD_COUNT 8
#define INTERSECT_FLAG unsigned char  //do not set CHILD_COUNT biger then the bit of the INTERSECT_FLAG

#include<iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct fVertex
{
	float x, y, z;

	__host__ __device__
		bool operator<(const fVertex a) const
	{
		return x < a.x;
	}
};

struct face
{
	int v[3];
};

struct obj
{
	int vertexCount;
	fVertex *vertexArray;

	int faceCount;
	face *faceArray;
};

struct box
{
	//the max value of x,y,z
	float xMax;
	float yMax;
	float zMax;

	//the min value of x,y,z
	float xMin;
	float yMin;
	float zMin;
};

struct childIndex
{
	int index[CHILD_COUNT];
};

struct node
{
	box *b;
	childIndex *child;
};

struct rtree
{
	int layer;
	int nodeCount;
	node n;
	box * leaf;
};

struct childNode
{
	int nodeIndex;
	int boxIndex;
};

struct searchResult
{
	int * leafList;
	int * boxId;
	int length;
};


void debugBox(int length, box *b, int start, int end);

void printBox(box b);

void spendTime();

void cudaEventInit();