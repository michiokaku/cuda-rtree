#include "searchRtree.cuh"
#include "build_rtree.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define intersectTest(a,b) (!((a.xMin > b.xMax) || (b.xMin > a.xMax) || (a.yMin > b.yMax) || (b.yMin > a.yMax) || (a.zMin > b.zMax) || (b.zMin > a.zMax)))

__device__ void pBox(box b)
{
	printf("xMax:%f ; xMin:%f ; yMax:%f ; yMin:%f ; zMax:%f ; zMin:%f ;\n", b.xMax, b.xMin, b.yMax, b.yMin, b.zMax, b.zMin);
}

__global__ void FirstSearchRtreeKernel(box *searchBox, rtree r,INTERSECT_FLAG *inFlag,int *intersectChildCount,int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		box sb = searchBox[tid];
		box cb;
		int index;

		INTERSECT_FLAG flag = 0;
		int icc = 0	;

		for (int i = 0;i < CHILD_COUNT;i++)
		{
			index = r.n.childIndex[getChild(0,i,r.nodeCount)];
			if (index > 0) {
				
				cb = r.n.b[index];
				if(intersectTest(sb, cb))
				{
					flag |= 1 << i;
					icc ++;
				}
			}
		}

		intersectChildCount[tid] = icc;
		inFlag[tid] = flag;
	}
}

__global__ void FirstBuildChildList(int *perfixSum, INTERSECT_FLAG *inFlag,int *childList,int * boxId,rtree r,int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		int offset = 0;
		if (tid > 0)offset = perfixSum[tid - 1];
		INTERSECT_FLAG flag = inFlag[tid];
		int counter = 0;

		//load child node
		for (int i = 0;i < CHILD_COUNT;i++)
		{
			if (((flag >> i) & 1) == 1)
			{
				childList[offset + counter] = r.n.childIndex[getChild(0, i, r.nodeCount)];
				boxId[offset + counter] = tid;
				counter++;
			}
		}
	}
}

__global__ void SearchRtreeKernel(box *searchBox, int *boxId, rtree r, int *nodeList, INTERSECT_FLAG *inFlag, int *intersectChildCount, int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		box sb = searchBox[boxId[tid]];
		box cb;
		int index;

		INTERSECT_FLAG flag = 0;
		int icc = 0;

		for (int i = 0;i < CHILD_COUNT;i++)
		{
			index = r.n.childIndex[getChild(nodeList[tid], i, r.nodeCount)];
			if (index > 0) {

				cb = r.n.b[index];
				if (intersectTest(sb, cb))
				{
					flag |= 1 << i;
					icc++;
				}
			}
		}

		intersectChildCount[tid] = icc;
		inFlag[tid] = flag;
	}
}

__global__ void BuildChildList(int *perfixSum, INTERSECT_FLAG *inFlag,int *nodeList ,int *childList,int * oldBoxId, int * boxId, rtree r, int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		int offset = 0;
		if (tid > 0)offset = perfixSum[tid - 1];
		INTERSECT_FLAG flag = inFlag[tid];
		int counter = 0;
		int obi = oldBoxId[tid];

		//load child node
		for (int i = 0;i < CHILD_COUNT;i++)
		{
			if (((flag >> i) & 1) == 1)
			{
				childList[offset + counter] = r.n.childIndex[getChild(nodeList[tid], i, r.nodeCount)];
				boxId[offset + counter] = obi;
				counter++;
			}
		}
	}
}

__global__ void LastSearchRtreeKernel(box *searchBox, int *boxId, rtree r, int *nodeList, INTERSECT_FLAG *inFlag, int *intersectChildCount, int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		box sb = searchBox[boxId[tid]];
		box cb;
		int index;

		INTERSECT_FLAG flag = 0;
		int icc = 0;

		for (int i = 0;i < CHILD_COUNT;i++)
		{
			index = r.n.childIndex[getChild(nodeList[tid], i, r.nodeCount)];
			if (index > 0) {

				cb = r.leaf[index];
				if (intersectTest(sb, cb))
				{
					flag |= 1 << i;
					icc++;
				}
			}
		}

		intersectChildCount[tid] = icc;
		inFlag[tid] = flag;
	}
}

__global__ void LastBuildChildList(int *perfixSum, INTERSECT_FLAG *inFlag, int *nodeList, int *childList, int * oldBoxId, int * boxId, rtree r, int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		int offset = 0;
		if (tid > 0)offset = perfixSum[tid - 1];
		INTERSECT_FLAG flag = inFlag[tid];
		int counter = 0;
		int obi = oldBoxId[tid];

		//load child node
		for (int i = 0;i < CHILD_COUNT;i++)
		{
			if (((flag >> i) & 1) == 1)
			{
				childList[offset + counter] = r.n.childIndex[getChild(nodeList[tid], i, r.nodeCount)];
				boxId[offset + counter] = obi;
				counter++;
			}
		}
	}
}

__global__ void LastBuildChildListNode(int *perfixSum, INTERSECT_FLAG *inFlag, int *nodeList, int *childList, int * oldBoxId, int * boxId, rtree r, int length)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;

	if (tid < length)
	{
		int parentId = tid / CHILD_COUNT;

		int offset = 0;
		if (tid > 0)offset = perfixSum[tid - 1];
		INTERSECT_FLAG flag = inFlag[tid];
		int counter = 0;
		int obi = oldBoxId[tid];

		//load child node
		for (int i = 0;i < CHILD_COUNT;i++)
		{
			if (((flag >> i) & 1) == 1)
			{
				childList[offset + counter] = r.n.childIndex[getChild(nodeList[tid], i, r.nodeCount)];
				boxId[offset + counter] = obi;
				counter++;
			}
		}
	}
}

void debugInt(int *a, int len)
{
	int *b = (int*)malloc(len * sizeof(int));

	cudaMemcpy(b, a , len*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < len;i++)
	{
		printf("a[%d] = %d\n", i, b[i]);
	}
}

int getChildLength(int *intersectChildCount,int len)
{
	int length;

	cudaMemcpy(&length, intersectChildCount+(len-1), sizeof(int), cudaMemcpyDeviceToHost);

	return length;
}

void boxidDebug(int *boxid, int length)
{
	int *bhost = (int*)malloc(length * sizeof(int));
	cudaMemcpy(bhost, boxid, length * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0;i < length;i++)
	{
		std::cout<<"i:"<<i<<" = "<<bhost[i]<<"  ";
	}
	std::cout << "\n";
	free(bhost);
}

void iccDebug(int *intersectChildCount, int length)
{
	int *bhost = (int*)malloc(length * sizeof(int));
	cudaMemcpy(bhost, intersectChildCount, length * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0;i < length;i++)
	{
		std::cout << "i:" << i << " = " << bhost[i] << "  ";
	}
	std::cout << "\n";
	free(bhost);
}

int searchLeafLayer(int nodeLength, int * &nodeList, box *searchBox, int * &boxId, rtree r)
{
	INTERSECT_FLAG * inFlag;
	int *intersectChildCount;

	//std::cout << nodeLength << "\n";

	cudaMalloc((void**)&inFlag, nodeLength * sizeof(INTERSECT_FLAG));
	cudaMalloc((void**)&intersectChildCount, nodeLength * sizeof(int));
	LastSearchRtreeKernel << <GetBlockCount(nodeLength), THREAD_PER_BLOCK >> >(searchBox, boxId, r, nodeList, inFlag, intersectChildCount, nodeLength);
	//boxidDebug(nodeList, nodeLength);
	//iccDebug(intersectChildCount, nodeLength);

	thrust::device_ptr<int> sum_ptr(intersectChildCount);
	thrust::inclusive_scan(sum_ptr, sum_ptr + nodeLength, sum_ptr);


	int childLength = getChildLength(intersectChildCount, nodeLength);
	std::cout << childLength << "\n";

	int *childList, *newBoxId;
	cudaMalloc((void**)&childList, childLength * sizeof(int));
	cudaMalloc((void**)&newBoxId, childLength * sizeof(int));

	LastBuildChildList << <GetBlockCount(nodeLength), THREAD_PER_BLOCK >> >(intersectChildCount, inFlag, nodeList, childList, boxId, newBoxId, r, nodeLength);

	//boxidDebug(childList, childLength);
	//free memery
	cudaFree(intersectChildCount);
	cudaFree(inFlag);

	cudaFree(nodeList);
	cudaFree(boxId);

	boxId = newBoxId;
	nodeList = childList;

	return childLength;
}

int searchLayer(int nodeLength,int * &nodeList,box *searchBox,int * &boxId,rtree r)
{
	INTERSECT_FLAG * inFlag;
	int *intersectChildCount;

	//std::cout << nodeLength << "\n";

	cudaMalloc((void**)&inFlag, nodeLength * sizeof(INTERSECT_FLAG));
	cudaMalloc((void**)&intersectChildCount, nodeLength * sizeof(int));
	SearchRtreeKernel << <GetBlockCount(nodeLength), THREAD_PER_BLOCK >> >(searchBox, boxId, r, nodeList, inFlag, intersectChildCount, nodeLength);
	//boxidDebug(nodeList, nodeLength);
	//iccDebug(intersectChildCount, nodeLength);

	thrust::device_ptr<int> sum_ptr(intersectChildCount);
	thrust::inclusive_scan(sum_ptr, sum_ptr + nodeLength, sum_ptr);
	int childLength = getChildLength(intersectChildCount, nodeLength);
	std::cout << childLength<<"\n";

	int *childList, *newBoxId;
	cudaMalloc((void**)&childList, childLength * sizeof(int));
	cudaMalloc((void**)&newBoxId, childLength * sizeof(int));

	BuildChildList << <GetBlockCount(nodeLength), THREAD_PER_BLOCK >> >(intersectChildCount, inFlag, nodeList, childList, boxId, newBoxId, r, nodeLength);

	//boxidDebug(childList, childLength);
	//free memery
	cudaFree(intersectChildCount);
	cudaFree(inFlag);

	cudaFree(nodeList);
	cudaFree(boxId);

	boxId = newBoxId;
	nodeList = childList;

	return childLength;
}

searchResult searchRtree(box *searchBox,int boxCount,rtree r)
{
	int intersectTimes = 0;

	INTERSECT_FLAG * inFlag;
	int *intersectChildCount;
	cudaMalloc((void**)&inFlag, boxCount * sizeof(INTERSECT_FLAG));
	cudaMalloc((void**)&intersectChildCount, boxCount * sizeof(int));

	box * dev_searchBox;
	cudaMalloc((void**)&dev_searchBox, boxCount * sizeof(box));
	cudaMemcpy(dev_searchBox, searchBox, boxCount * sizeof(box), cudaMemcpyHostToDevice);
	
	//firts search is diferent for others
	FirstSearchRtreeKernel << <GetBlockCount(boxCount), THREAD_PER_BLOCK >> > (dev_searchBox, r, inFlag, intersectChildCount, boxCount);

	//iccDebug(intersectChildCount, boxCount);

	thrust::device_ptr<int> sum_ptr(intersectChildCount);
	thrust::inclusive_scan(sum_ptr, sum_ptr + boxCount, sum_ptr);
	int childLength = getChildLength(intersectChildCount, boxCount);
	int *childList,*boxId;
	cudaMalloc((void**)&childList, childLength * sizeof(int));
	cudaMalloc((void**)&boxId, childLength * sizeof(int));
	FirstBuildChildList << <GetBlockCount(boxCount), THREAD_PER_BLOCK >> >(intersectChildCount, inFlag, childList, boxId, r, boxCount);
	//free memery
	cudaFree(intersectChildCount);
	cudaFree(inFlag);

	std::cout << childLength << "\n";
	intersectTimes += childLength;
	//
	for (int i = 1;i < r.layer-1;i++)
	{
		if (childLength < 1)break;
		childLength = searchLayer(childLength, childList, dev_searchBox, boxId, r);
		intersectTimes += childLength;
	}

	childLength = searchLeafLayer(childLength, childList, dev_searchBox, boxId, r);
	intersectTimes += childLength;
	std::cout << "intersectTimes is " << intersectTimes << "\n";

	searchResult sr;
	sr.boxId = boxId;
	sr.length = childLength;
	sr.leafList = childList;

	return sr;
}