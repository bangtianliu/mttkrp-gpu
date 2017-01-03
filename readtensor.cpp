#include "readtensor.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using std::ifstream;
using std::istringstream;
using std::endl;
using std::cout;
using std::string;

/**
 * @param
 * @param
 * @param
 * @param
 * @return
 */
int precess(int &dim_i,int &dim_j, int &dim_k, char *file)
{
	ifstream in_stream(file);
	string line;
	int total_nnz=0;		
	int i,j,k;
	float value;
	while(getline(in_stream, line)){
		istringstream liness(line);
		liness >> i >> j >> k >> value;
		dim_i = i > dim_i? i : dim_i;
		dim_j = j > dim_j? j : dim_j;
		dim_k = k > dim_k? k : dim_k;
		total_nnz++; 
	}

	return total_nnz;
}


void tensor_malloc(tensor *data, int nnz)
{
	*data=(tensor)malloc(sizeof(item)*nnz);
}

void tensor_free(tensor data)
{
	free(data);
}

int readtensor(tensor data, char *file)
{
	ifstream in_stream(file);
	string line;
	int index=0;
	// int i,j,k;
	cout<<"in readtensor"<<endl;
	while(getline(in_stream, line)){
		// cout<<"in readtensor"<<endl;
		istringstream liness(line);
		liness >> data[index].coord[0] >> data[index].coord[1] >> data[index].coord[2] >> data[index].val;
		// cout<<"in readtensor2"<<endl;
		index++;

		if(index % 1000000 ==0){
			cout << "Total lines read: " << index << endl;
		}
	}

	return index;
}