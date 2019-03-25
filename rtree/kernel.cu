
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

box* genBox(int length)
{
	box *b = (box*)malloc(length * sizeof(box));

	for (int i = 0;i < length;i++)
	{
		box rb;
		rb.xMax = random(1.0f, i);
		rb.xMin = random(rb.xMax, i + length);

		rb.yMax = random(1.0f, i+(length*2));
		rb.yMin = random(rb.yMax, i + (length * 3));

		rb.zMax = random(1.0f, i+ (length * 4));
		rb.zMin = random(rb.zMax, i + (length * 5));

		b[i] = rb;
	}

	return b;
}

int main()
{
	rtree r = buildRtree();

	int length = 10;
	box * b = genBox(length);

	searchRtree(b, length, r);

	//system("pause");

	//cudaDeviceReset();
	return 0;
}

