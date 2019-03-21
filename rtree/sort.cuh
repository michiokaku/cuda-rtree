#pragma once
#include "cub\cub.cuh"
#include "dataStruct.h"

box * sort_cub(unsigned int * key, box * value, int length);