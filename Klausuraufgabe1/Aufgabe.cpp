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

int praefixsumme(cl_int* input, cl_int* output_b, cl_int* output_c, cl_int* output_d, int size, OpenCLMgr& mgr)
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
		status = clEnqueueWriteBuffer(mgr.commandQueue, aBuffer, CL_TRUE, size * sizeof(cl_int), (clsize-size)*sizeof(cl_int), &tmp, 0, NULL, NULL);
		CHECK_SUCCESS("Error: writing buffer!")
	}
	cl_mem bBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);
	cl_mem cBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);
	cl_mem dBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	status = clSetKernelArg(mgr.praefixsumme256_kernel, 0, sizeof(cl_mem), (void *)&aBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 1, sizeof(cl_mem), (void *)&bBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 2, sizeof(cl_mem), (void *)&cBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 3, sizeof(cl_mem), (void *)&dBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	size_t global_work_size[1] = { clsize };
	size_t local_work_size[1] = { fields };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.praefixsumme256_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	status = clEnqueueReadBuffer(mgr.commandQueue, bBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output_b, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")
	status = clEnqueueReadBuffer(mgr.commandQueue, cBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output_c, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")
	status = clEnqueueReadBuffer(mgr.commandQueue, dBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output_d, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")

	status = clReleaseMemObject(aBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(bBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(cBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
	status = clReleaseMemObject(dBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}

cl_int summe(cl_int* inputB, cl_int* inputD, cl_int* output_e, int size, OpenCLMgr& mgr)
{
	cl_int status;
	int result;

	int fields = BLOCK_SIZE;
	int clsize = (size + (fields - 1)) / fields * fields;  // das Vielfache von 256 (256,512,1024,1280,...)

	cl_mem inputBBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int), NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputBBuffer, CL_TRUE, 0, size * sizeof(cl_int), inputB, 0, NULL, NULL);
	cl_mem inputDBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int), NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, inputDBuffer, CL_TRUE, 0, size * sizeof(cl_int), inputD, 0, NULL, NULL);
	cl_mem outputBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	status = clSetKernelArg(mgr.summe_kernel, 0, sizeof(cl_mem), (void*)&inputBBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.summe_kernel, 1, sizeof(cl_mem), (void*)&inputDBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")
	status = clSetKernelArg(mgr.summe_kernel, 2, sizeof(cl_mem), (void*)&outputBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	size_t global_work_size[1] = { clsize };
	size_t local_work_size[1] = { fields };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.summe_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	status = clEnqueueReadBuffer(mgr.commandQueue, outputBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output_e, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading buffer!")

	// release buffers
	status = clReleaseMemObject(inputBBuffer);
	status = clReleaseMemObject(inputDBuffer);
	status = clReleaseMemObject(outputBuffer);

	return SUCCESS;
}

int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	// Initial input,output for the host and create memory objects for the kernel
	int size = 20;
	cl_int* input = new cl_int[size];
	cl_int* output_b = new cl_int[size];
	cl_int* output_c = new cl_int[size];
	cl_int* output_d = new cl_int[size];

	for (int i = 0; i < size; i++)
		input[i] = 1;

	// call function
	praefixsumme(input, output_b, output_c, output_d, size, mgr);

	cl_int* output_e = new cl_int[size];
	summe(output_b, output_d, output_e, size, mgr);

	for (int i = 0; i < size; i++)
		cout << output_e[i] << "\n";

	delete[] input;
	delete[] output_b;
	delete[] output_c;
	delete[] output_d;
	delete[] output_e;

	std::cout << "Passed!\n";
	return SUCCESS;
}
