#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "build_rtree.cuh"
#include "readObj.h"
#include "dataStruct.h"

#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "sort.cuh"
#include  <direct.h>

#define GETVERTEX(o,f,j) o.vertexArray[f.v[j]];
#define THREAD_PER_BLOCK 256
#define SCALE 0.1 //scale the object;

int GetBlockCount(int threadCount) {

	int blockCount = threadCount / THREAD_PER_BLOCK;

	if ((threadCount%THREAD_PER_BLOCK) != 0)
	{
		++blockCount;
	}

	return blockCount;
}

__global__ void scaleObj(fVertex *fv,int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		fVertex f = fv[tid];
		f.x *= SCALE;
		f.y *= SCALE;
		f.z *= SCALE;

		fv[tid] = f;
	}
}

__global__ void buildBoxKernel(box *b, obj o, fVertex *midpoint)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < o.faceCount)
	{

		//get vertex from vertex
		face f = o.faceArray[tid];
		fVertex v1 = GETVERTEX(o, f, 0);
		fVertex v2 = GETVERTEX(o, f, 1);
		fVertex v3 = GETVERTEX(o, f, 2);

		//get max and min value of x,y,z
		float xMax = v1.x;
		xMax = fmaxf(v2.x, xMax);
		xMax = fmaxf(v3.x, xMax);

		float xMin = v1.x;
		xMin = fminf(v2.x, xMin);
		xMin = fminf(v3.x, xMin);

		float yMax = v1.y;
		yMax = fmaxf(v2.y, yMax);
		yMax = fmaxf(v3.y, yMax);

		float yMin = v1.y;
		yMin = fminf(v2.y, yMin);
		yMin = fminf(v3.y, yMin);

		float zMax = v1.z;
		zMax = fmaxf(v2.z, zMax);
		zMax = fmaxf(v3.z, zMax);

		float zMin = v1.z;
		zMin = fminf(v2.z, zMin);
		zMin = fminf(v3.z, zMin);

		//build box on the register
		box regBox;
		regBox.xMax = xMax;
		regBox.xMin = xMin;

		regBox.yMax = yMax;
		regBox.yMin = yMin;

		regBox.zMax = zMax;
		regBox.zMin = zMin;


		//return regBox
		b[tid] = regBox;

		//count minpoint
		fVertex mp;
		mp.x = (xMax + xMin) / 2.0f;
		mp.y = (yMax + yMin) / 2.0f;
		mp.z = (zMax + zMin) / 2.0f;

		//return midpoint
		midpoint[tid] = mp;
	}
}

void buildBox(box *b, obj o, fVertex *midpoint)
{
	scaleObj << <GetBlockCount(o.vertexCount), THREAD_PER_BLOCK >> > (o.vertexArray, o.vertexCount);
	buildBoxKernel << <GetBlockCount(o.faceCount), THREAD_PER_BLOCK >> > (b, o, midpoint);

	//box *host_b = (box*)malloc(o.faceCount * sizeof(box));
	//cudaMemcpy(host_b, b, o.vertexCount * sizeof(fVertex), cudaMemcpyDeviceToHost);
}

__device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z)
{
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__global__ void CountZorder(unsigned int * zOrder, fVertex * point, int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		fVertex p = point[tid];
		zOrder[tid] = morton3D(p.x, p.y, p.z);
	}
}

__global__ void test(unsigned int * zOrder)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
}

void SortBox(box *b, fVertex *midpoint, int length)
{
	//build zOrder value of the midpoint
	unsigned int * zOrder;
	cudaMalloc((void**)&zOrder, length * sizeof(unsigned int));

	CountZorder << <GetBlockCount(length), THREAD_PER_BLOCK >> > (zOrder, midpoint, length);

	b = sort_cub(zOrder, b, length);
}

__device__ box BigerBox(box a, box b)
{
	a.xMax = fmaxf(a.xMax, b.xMax);
	a.xMin = fmaxf(a.xMin, b.xMin);

	a.yMax = fmaxf(a.yMax, b.yMax);
	a.yMin = fmaxf(a.yMin, b.yMin);

	a.zMax = fmaxf(a.zMax, b.zMax);
	a.zMin = fmaxf(a.zMin, b.zMin);
}

__global__ void FirstMergeBoxKernel(node *n ,box *b,int offset,int parentLength,int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < parentLength)
	{
		int childIndex = tid * CHILD_COUNT;
		box rb = b[childIndex];//the box of node
		node rn;
		rn.child[0] = childIndex;

		//count the number of children
		int childNum = CHILD_COUNT;
		if (tid == parentLength - 1)childNum = length%CHILD_COUNT;

		box cb;
		for (int i = 1;i < childNum;i++)
		{
			rn.child[i] = childIndex + i;
			cb = b[childIndex + i];
			rb = BigerBox(rb, cb);
		}

		//return node
		rn.b = rb;
		n[offset + tid] = rn;
	}
}

rtree mergeBox(box *b,int length)
{
	rtree r;

	r.nodeCount = 0;
	r.layer = 0;
	//Do not count the last layer

	int len = length;
	do
	{
		len = (len + CHILD_COUNT - 1) / CHILD_COUNT;
		r.nodeCount += len;
		r.layer++;
	} 
	while (len > 1);

	cudaMalloc((void**)&r.n, r.nodeCount * sizeof(unsigned int));

	len = length;
	len = (len + CHILD_COUNT - 1) / CHILD_COUNT;
	int offset = r.nodeCount - len;
	//FirstMergeBoxKernel << <GetBlockCount(len), THREAD_PER_BLOCK >> > (r.n, b, offset, len,length);

	for (int i = 0;i < r.layer;i++)
	{

	}

	return r;
}

void buildRtree()
{
	//read obj
	obj o = ReadObj("C:\\Users\\chenxiyu\\Documents\\Visual Studio 2015\\Projects\\rtree\\media\\dragon.obj");

	//build obj on the device
	obj dev_o;
	dev_o.faceCount = o.faceCount;
	dev_o.vertexCount = o.vertexCount;
	cudaMalloc((void**)&dev_o.vertexArray, dev_o.vertexCount * sizeof(fVertex));
	cudaMalloc((void**)&dev_o.faceArray, dev_o.faceCount * sizeof(face));

	//copy the data of obj witch from host to device
	cudaMemcpy(dev_o.vertexArray, o.vertexArray, dev_o.vertexCount * sizeof(fVertex), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_o.faceArray, o.faceArray, dev_o.vertexCount * sizeof(face), cudaMemcpyHostToDevice);

	//build box for each face
	fVertex *dev_midpoint;      //build the midpoint of each box
	box *dev_box;               //build the box on the device
	cudaMalloc((void**)&dev_midpoint, dev_o.faceCount * sizeof(fVertex));
	cudaMalloc((void**)&dev_box, o.faceCount * sizeof(box));
	buildBox(dev_box, dev_o, dev_midpoint);

	//sort box by the zorder
	SortBox(dev_box, dev_midpoint, o.faceCount);

	//mergeBox(dev_box, o.faceCount);

	//cudaFree(dev_midpoint);
}