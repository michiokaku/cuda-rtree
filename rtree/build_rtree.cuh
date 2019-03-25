#pragma once

#include "dataStruct.h"

#define GETVERTEX(o,f,j) o.vertexArray[f.v[j]];
#define THREAD_PER_BLOCK 256
#define SCALE 0.05 //scale the object;

rtree buildRtree();
int GetBlockCount(int threadCount);