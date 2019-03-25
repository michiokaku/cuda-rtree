#include "searchRtree.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define intersectTest(a,b) (((a.xMin > b.xMax) || (b.xMin > a.xMax)) && ((a.yMin > b.yMax) || (b.yMin > a.yMax)) && ((a.zMin > b.zMax) || (b.zMin > a.zMax)))


__global__ void FirstSearchRtreeKernel(box *searchBox, node n,INTERSECT_FLAG *inFlag,int *intersectChildCount,int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		box sb = searchBox[tid];
		childIndex ci = n.child[0];
		INTERSECT_FLAG flag = 0;
		bool rig;
		int icc = 0;

		for (int i = 0;i < CHILD_COUNT;i++)
		{
			rig = intersectTest(sb, n.b[ci.index[i]]);
			flag |= rig;
			icc += rig;
		}

		intersectChildCount[tid] = icc;
		inFlag[tid] = flag;
	}
}

void searchRtree(box *searchBox,int boxCount,rtree r)
{
	for (int i = 0;i < r.layer-1;i++)
	{
		
	}
}