#include "searchRtree.cuh"
#include "build_rtree.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define intersectTest(a,b) (((a.xMin > b.xMax) || (b.xMin > a.xMax)) && ((a.yMin > b.yMax) || (b.yMin > a.yMax)) && ((a.zMin > b.zMax) || (b.zMin > a.zMax)))

__device__ bool interTest(box a, box b)
{
	bool flag = (a.xMin > b.xMax);

	return flag;
}

__global__ void FirstSearchRtreeKernel(box *searchBox, node n,INTERSECT_FLAG *inFlag,int *intersectChildCount,int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		box sb = searchBox[tid];
		childIndex ci = n.child[0];
		box cb;
		int index;

		INTERSECT_FLAG flag = 0;
		bool rig = 1;
		int icc = 0;

		for (int i = 0;i < CHILD_COUNT;i++)
		{
			index = ci.index[i];
			if (index > 0) {
				
				cb = n.b[0];
				if(intersectTest(sb, cb))
				{
					flag |= 1 << index;
					icc += index;
				}
			}
		}

		intersectChildCount[tid] = icc;
		inFlag[tid] = flag;
	}
}

void searchRtree(box *searchBox,int boxCount,rtree r)
{
	INTERSECT_FLAG * inFlag;
	int *intersectChildCount;
	cudaMalloc((void**)&inFlag, boxCount * sizeof(INTERSECT_FLAG));
	cudaMalloc((void**)&intersectChildCount, boxCount * sizeof(int));

	box * dev_searchBox;
	cudaMalloc((void**)&dev_searchBox, boxCount * sizeof(box));
	cudaMemcpy(dev_searchBox, searchBox, boxCount, cudaMemcpyHostToDevice);

	FirstSearchRtreeKernel << <GetBlockCount(boxCount), THREAD_PER_BLOCK >> > (dev_searchBox, r.n, inFlag, intersectChildCount, boxCount);

	for (int i = 0;i < r.layer-1;i++)
	{
		
	}
}