#include<stdio.h>
#include<stdlib.h>
#include"readObj.h"
#include"dataStruct.h"

obj ReadObj(char * fileName)
{
	FILE *fp;
	char str[200];
	fp = fopen(fileName, "rt");
	if (fp == NULL)printf("\nCan not open obj File!\n");

	//Get the count of the point and face
	obj o;
	o.vertexCount = 0;
	o.faceCount = 0;
	while (fgets(str, 200, fp) != NULL)
	{
		if (str[0] == 'v')
		{
			o.vertexCount++;
		}
		else if (str[0] == 'f')
		{
			o.faceCount++;
		}
	}

	//read obj
	rewind(fp);
	o.faceArray = (face*)malloc(o.faceCount * sizeof(face));
	o.vertexArray = (fVertex*)malloc(o.vertexCount * sizeof(fVertex));

	int vertexNum = 0;
	int faceNum = 0;

	while (fgets(str, 200, fp) != NULL)
	{
		if (str[0] == 'v')
		{
			sscanf(str, "v %f %f %f", &o.vertexArray[vertexNum].x, &o.vertexArray[vertexNum].y, &o.vertexArray[vertexNum].z);
			vertexNum++;
		}
		else if (str[0] == 'f')
		{
			sscanf(str, "f %d %d %d", &o.faceArray[faceNum].v[0], &o.faceArray[faceNum].v[1], &o.faceArray[faceNum].v[2]);
			faceNum++;
		}
	}

	return o;
}