#pragma once

#define CHILD_COUNT 8

struct fVertex
{
	float x, y, z;
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

struct node
{
	box b;
	int child[CHILD_COUNT];
};

struct rtree
{
	int layer;
	int nodeCount;
	node *n;
};