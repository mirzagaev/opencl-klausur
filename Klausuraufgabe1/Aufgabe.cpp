// Klausuraufgabe1.cpp : Diese Datei enthält die Funktion "main".
// Hier beginnt und endet die Ausführung des Programms.

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#include "OpenCLMgr.h"

#define SUCCESS 0
#define FAILURE 1

using namespace std;

cl_int praefixsummeRek(cl_mem aBuffer, cl_int* output, int size, OpenCLMgr& mgr)
{
	cl_int status;
	int fields = 256;
	size_t outsize = size / fields;
	int clsize = (outsize + (fields-1)) / fields * fields;  // das Vielfache von 256 (256,512,1024,1280,...)

	cout << "\nsize: " << size << "\n";
	cout << "outsize: " << outsize << "\n";
	cout << "clsize: " << clsize << "\n";

	// create OpenCl buffer for output
	cl_mem bBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_WRITE, clsize * sizeof(cl_int), NULL, NULL);

	// Set kernel arguments.
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 0, sizeof(cl_mem), (void*)&aBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 1!")
	status = clSetKernelArg(mgr.praefixsumme256_kernel, 1, sizeof(cl_mem), (void*)&bBuffer);
	CHECK_SUCCESS("Error: setting kernel argument 2!")

	// Run the kernel.
	size_t global_work_size[1] = { fields };
	size_t local_work_size[1] = { fields };
	status = clEnqueueNDRangeKernel(mgr.commandQueue, mgr.praefixsumme256_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CHECK_SUCCESS("Error: enqueuing kernel!")

	// get resulting array
	status = clEnqueueReadBuffer(mgr.commandQueue, bBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), output, 0, NULL, NULL);
	CHECK_SUCCESS("Error: reading bBuffer!")

	if (outsize > 1) {
		if (size < clsize) {
			// unintialisierte Elemente [0-255] 0 setzen, um Berechnung zu ermöglichen
			cl_int tmp[256] = { 0 };
			status = clEnqueueWriteBuffer(mgr.commandQueue, aBuffer, CL_TRUE, size * sizeof(cl_int), (clsize-size)*sizeof(cl_int), &tmp, 0, NULL, NULL);
			CHECK_SUCCESS("Error: writing buffer!")
			cout << "aBuffer override #58\n";
		}

		praefixsummeRek(aBuffer, output, clsize, mgr);
		cout << "more then 256 elements" << "\n";
	}
	
	status = clReleaseMemObject(bBuffer);
	CHECK_SUCCESS("Error: releasing buffer!")
}

// size of arrays must be exactly 256
cl_int praefixsumme(cl_int* input, cl_int* output, int size, OpenCLMgr& mgr)
{
	cl_int status;
	int fields = 256;
	int clsize = (size + (fields-1)) / fields * fields;  // das Vielfache von 256 (256,512,1024,1280,...)
	
	cout << "\nsize: " << size << "\n";
	cout << "clsize: " << clsize << "\n";

	// create OpenClinput buffer
	cl_mem aBuffer = clCreateBuffer(mgr.context, CL_MEM_READ_ONLY, clsize * sizeof(cl_int),NULL, NULL);
	status = clEnqueueWriteBuffer(mgr.commandQueue, aBuffer, CL_TRUE, 0, clsize * sizeof(cl_int), input, 0, NULL, NULL);
	CHECK_SUCCESS("Error: writing aBuffer!")
	if (size < clsize) {
		// unintialisierte Elemente [0-255] 0 setzen, um Berechnung zu ermöglichen
		cl_int tmp[256] = { 0 };
		status = clEnqueueWriteBuffer(mgr.commandQueue, aBuffer, CL_TRUE, size * sizeof(cl_int), (clsize-size)*sizeof(cl_int), &tmp, 0, NULL, NULL);
		CHECK_SUCCESS("Error: writing buffer!")
		cout << "aBuffer override #88\n";
	}

	praefixsummeRek(aBuffer, output, clsize, mgr);

	// release buffers
	status = clReleaseMemObject(aBuffer);		
	CHECK_SUCCESS("Error: releasing buffer!")

	return SUCCESS;
}

int main(int argc, char* argv[])
{
	OpenCLMgr mgr;

	// Initial input,output for the host and create memory objects for the kernel
	int size = 510;
	cl_int* input = new cl_int[size];
	cl_int* output = new cl_int[size];
	for (int i = 0; i < size; i++)
		input[i] = 1;

	praefixsumme(input, output, size, mgr);

	delete[] input;
	delete[] output;

	std::cout << "Passed!\n";
	
	return 1;
}