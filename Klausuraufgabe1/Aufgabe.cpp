#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1

using namespace std;

#include "OpenCLMgr.h"

// size of arrays must be exactly 256
int praefixsumme(cl_int *input, cl_int *output, int size, OpenCLMgr& mgr)
{
	cl_int status;

	int fields = 8;
	int clsize = (size + (fields - 1)) / fields * fields;  // das Vielfache von 256 (256,512,1024,1280,...)

	// create OpenClinput buffer
	cl_mem inputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int),NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, 0, size * sizeof(cl_int), input, 0, NULL, NULL);
	CHECK_SUCCESS("Error: writing buffer!")
		if (size < clsize) {
		cl_int tmp[8] = { 0 };
		status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, size * sizeof(cl_int), (clsize-size)*sizeof(cl_int), &tmp, 0, NULL, NULL);
		CHECK_SUCCESS("Error: writing buffer!")
	}

	// create OpenCl buffer for output
	cl_mem outputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	// Set kernel arguments.
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	// Run the kernel.
	size_t global_work_size[1] = { clsize };
	size_t local_work_size[1] = { fields };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.praefixsumme256_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	// get resulting array
	status = clEnqueueReadBuffer(mgr.commandQueue, outputBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")

	// release buffers
	status = clReleaseMemObject(inputBuffer);		
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(outputBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}


int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	// Initial input,output for the host and create memory objects for the kernel
	int size = 13;
	cl_int *input = new cl_int[size];
	cl_int *output = new cl_int[size];

	for (int i=0 ; i<size ; i++)
		input[i] = 1;

	// call function
	praefixsumme(input, output, size, mgr);
	
	delete[] input;
	delete[] output;

	std::cout<<"Passed!\n";
	return SUCCESS;
}
