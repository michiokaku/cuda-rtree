#include "sort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>


box * sort_cub(unsigned int * key, box * value,int length)
{
	unsigned int *key_out;
	box *value_out;

	cudaMalloc((void**)&key_out, length * sizeof(unsigned int));
	cudaMalloc((void**)&value_out, length * sizeof(box));
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	cub::DeviceRadixSort::SortPairs<unsigned int, box>(d_temp_storage, temp_storage_bytes,
		key, key_out, value, value_out, length);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Run sorting operation
	cub::DeviceRadixSort::SortPairs<unsigned int, box>(d_temp_storage, temp_storage_bytes,
		key, key_out, value, value_out, length);

	cudaFree(key);
	cudaFree(value);
	cudaFree(key_out);

	cudaFree(d_temp_storage);

	return value_out;
}