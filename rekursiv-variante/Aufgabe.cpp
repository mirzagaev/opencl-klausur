#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1
#define BLOCK_SIZE 8

using namespace std;

#include "OpenCLMgr.h"

int praefixsummeCopy(cl_int* input, cl_int* b_output, cl_int* c_output, int size, OpenCLMgr& mgr)
{
	cl_int status;

	int fields = BLOCK_SIZE;
	int clsize = (size + (fields - 1)) / fields * fields;  // das Vielfache von 256 (256,512,1024,1280,...)

	// create OpenClinput buffer
	cl_mem inputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int), NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, 0, size * sizeof(cl_int), input, 0, NULL, NULL);
	CHECK_SUCCESS("Error: writing buffer!")
	if (size < clsize) {
		cl_int tmp[BLOCK_SIZE] = { 0 };
		status = clEnqueueWriteBuffer(mgr.commandQueue, inputBuffer, CL_TRUE, size * sizeof(cl_int), (clsize - size) * sizeof(cl_int), &tmp, 0, NULL, NULL);
		CHECK_SUCCESS("Error: writing buffer!")
	}
	cl_mem cBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, cBuffer, CL_TRUE, 0, size * sizeof(cl_int), c_output, 0, NULL, NULL);
	cl_mem dBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, dBuffer, CL_TRUE, 0, size * sizeof(cl_int), input, 0, NULL, NULL);

	cl_mem bBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	status = clSetKernelArg(mgr.praefixsumme256_kernel, 0, sizeof(cl_mem), (void*)&inputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 1, sizeof(cl_mem), (void*)&bBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 2, sizeof(cl_mem), (void*)&cBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	size_t global_work_size[1] = { clsize };
	size_t local_work_size[1] = { fields };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.praefixsumme256_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	status = clEnqueueReadBuffer(mgr.commandQueue, bBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), b_output, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")
	status = clEnqueueReadBuffer(mgr.commandQueue, cBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), c_output, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")

	status = clReleaseMemObject(inputBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(bBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(cBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(dBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}

cl_int praefixsummeRek(cl_mem aBuffer, cl_mem bBuffer, cl_int* output, int size, OpenCLMgr& mgr)
{
	cl_int status;

	size_t outsize = size / BLOCK_SIZE;
	size_t cloutsize = (outsize + (BLOCK_SIZE-1)) / BLOCK_SIZE * BLOCK_SIZE;

	cl_mem cBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, cloutsize * sizeof(cl_int), NULL, NULL);
	cl_mem dBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, cloutsize * sizeof(cl_int), NULL, NULL);

	status = clSetKernelArg(mgr.praefixsumme256_kernel, 0, sizeof(cl_mem), (void*)&aBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 1, sizeof(cl_mem), (void*)&bBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 2, sizeof(cl_mem), (void*)&cBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	size_t global_work_size[1] = { cloutsize };
	size_t local_work_size[1] = { BLOCK_SIZE };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.praefixsumme256_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	if (outsize == 1) {
		status = clEnqueueReadBuffer(mgr.commandQueue, bBuffer, CL_TRUE, 0, cloutsize * sizeof(cl_int), output, 0, NULL, NULL);
		CHECK_SUCCESS("Error: reading buffer!")
		//status = clEnqueueReadBuffer(mgr.commandQueue, cBuffer, CL_TRUE, 0, cloutsize * sizeof(cl_int), c_output, 0, NULL, NULL);
	}
	else {
		praefixsummeRek(cBuffer, dBuffer, output, cloutsize, mgr);
	}

	status = clReleaseMemObject(cBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(dBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}

cl_int praefixsumm(cl_int* input, cl_int* output, int size, OpenCLMgr& mgr)
{
	cl_int status;

	int fields = BLOCK_SIZE;
	int clsize = (size + (fields - 1)) / fields * fields;  // das Vielfache von 256 (256,512,1024,1280,...)

	// create OpenClinput buffer
	cl_mem aBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int), NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, aBuffer, CL_TRUE, 0, size * sizeof(cl_int), input, 0, NULL, NULL);
	CHECK_SUCCESS("Error: writing buffer!")
	if (size < clsize) {
		cl_int tmp[BLOCK_SIZE] = { 0 };
		status = clEnqueueWriteBuffer(mgr.commandQueue, aBuffer, CL_TRUE, size * sizeof(cl_int), (clsize - size) * sizeof(cl_int), &tmp, 0, NULL, NULL);
		CHECK_SUCCESS("Error: writing buffer!")
	}

	cl_mem bBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	praefixsummeRek(aBuffer, bBuffer, output, clsize, mgr);

	status = clReleaseMemObject(aBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(bBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}


int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	// Initial input,output for the host and create memory objects for the kernel
	int size = 20;
	cl_int* input = new cl_int[size];
	cl_int* output = new cl_int[size];

	for (int i = 0; i < size; i++)
		input[i] = 1;

	// call function
	praefixsumme(input, output, size, mgr);

	delete[] input;
	delete[] output;

	std::cout << "Passed!\n";
	return SUCCESS;
}
