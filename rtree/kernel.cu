
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "readObj.h"
#include "build_rtree.cuh"
#include "dataStruct.h"
#include "searchRtree.cuh"

float random(float range,int seed)
{
	int x = (13777 * seed + 3242) % 65536;

	for (int i = 0;i < 10;i++)
	{
		x = (17777 * x) % 65536;
	}

	float f = ((float)x) / 65536.0f;
	f = f*range;

	if (f > range)f = range;
	return f;
}

void seedAdd(int &seed)
{
	seed++;
	seed %= 65535;
}

box* genBox(int length)
{
	box *b = (box*)malloc(length * sizeof(box));

	int seed = 0;

	for (int i = 0;i < length;i++)
	{
		box rb;
		rb.xMax = random(1.0f, seed);
		seedAdd(seed);
		rb.xMin = random(rb.xMax, seed);
		seedAdd(seed);

		rb.yMax = random(1.0f, seed);
		seedAdd(seed);
		rb.yMin = random(rb.yMax, seed);
		seedAdd(seed);

		rb.zMax = random(1.0f, seed);
		seedAdd(seed);
		rb.zMin = random(rb.zMax, seed);
		seedAdd(seed);

		//rb.xMax = 1.0f;
		//rb.xMin = 0.0f;
		//rb.yMax = 1.0f;
		//rb.yMin = 0.0f;
		//rb.zMax = 1.0f;
		//rb.zMin = 0.0f;

		b[i] = rb;
	}

	return b;
}

int main()
{
	int length = 200000;
	box * b = genBox(length);

	cudaFree(0);

	rtree r = buildRtree();

	cudaEventInit();
	searchRtree(b, length, r);
	spendTime();

	//system("pause");

	//cudaDeviceReset();
	return 0;
}

