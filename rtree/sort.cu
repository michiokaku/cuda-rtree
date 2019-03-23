#include "sort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <iostream>


box * sort_thrust(unsigned int * key, box * value,int length)
{
	thrust::device_ptr<unsigned int> key_ptr(key);
	thrust::device_ptr<box>value_ptr(value);
	thrust::sort_by_key<thrust::device_ptr<unsigned int>, thrust::device_ptr<box>>(key_ptr, key_ptr + length, value_ptr);
	
	thrust::device_free(key_ptr);

	return value;
}