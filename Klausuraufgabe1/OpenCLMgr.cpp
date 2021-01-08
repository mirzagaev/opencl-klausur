#define SUCCESS 0
#define FAILURE 1

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

#include "OpenCLMgr.h"

OpenCLMgr::OpenCLMgr()
{
	context = 0;
	commandQueue = 0;
	program = 0;
	praefixsumme256_kernel = 0;
	summe_kernel = 0;

	valid = (init() == SUCCESS);
}

OpenCLMgr::~OpenCLMgr()
{
	cl_int status;

	// Gebe Ressourcen fur den Kernel frei.
	if (praefixsumme256_kernel) status = clReleaseKernel(praefixsumme256_kernel);
	if (summe_kernel) status = clReleaseKernel(summe_kernel);

	// Gebe Ressourcen wieder frei.
	if (program) status = clReleaseProgram(program);

	// Gebe Warteschlange wieder frei.
	if (commandQueue) status = clReleaseCommandQueue(commandQueue);

	// Gebe Ressourcen frei und losche den Kontext.
	if (context) status = clReleaseContext(context);
}

/* convert the kernel file into a string */
int OpenCLMgr::convertToString(const char* filename, std::string& s)
{
	size_t size;
	char* str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return FAILURE;
}

cl_int OpenCLMgr::init()
{
	cl_uint deviceNo = 1;

	// Getting platforms and choose an available one.
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	char* platforminfo;					// Speicherplatz fur die Ausgabe
	size_t platformlength;				// Anzahl der ausgegebenen Zeichen
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	CHECK_SUCCESS("Error: Getting platforms!")

	// For clarity, choose the first available platform.
	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];

		// Bestimme die Lange des Geratenamens.
		status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, 0, &platformlength);

		// Reserviere Speicher.
		platforminfo = (char*)malloc(platformlength);

		// Lese den Geratenamen ein.
		status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformlength, platforminfo, 0);
		std::cout << "CL_PLATFORM_NAME: " << platforminfo << "\n";
		free(platforms);
		CHECK_SUCCESS("Error: Getting platforms ids")
	}

	// Query devices and choose a GPU device if has one. Otherwise use the CPU as device.*/
	cl_uint				numDevices = 0;
	cl_device_id* devices;
	cl_device_id device;				// ID des abzufragenden Gerates
	char* devicename;					// Speicherplatz fur den Geratenamen
	size_t devicelength;				// Lange des Geratenamens
	cl_bool deviceavailable;			// Verfugbarkeit des Gerates
	cl_ulong devicememsize;				// Größe des globalen Speichers
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	CHECK_SUCCESS("Error: Getting device ids")
	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		CHECK_SUCCESS("Error: Getting number of cpu devices")
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		CHECK_SUCCESS("Error: Getting cpu device id")
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		device = devices[0];

		// Bestimme die Lange des Geratenamens.
		status = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, 0, &devicelength);

		// Reserviere Speicher.
		devicename = (char*)malloc(devicelength);

		// Lese den GerateInfos ein.
		status = clGetDeviceInfo(device, CL_DEVICE_NAME, devicelength, devicename, 0);
		status = clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &deviceavailable, 0);
		status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &devicememsize, 0);
		std::cout << "CL_DEVICE_NAME: " << devicename << "\n";
		std::cout << "CL_DEVICE_AVAILABLE: " << deviceavailable << "\n";
		std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE: " << devicememsize << "\n\n";
		CHECK_SUCCESS("Error: Getting gpu device id")
	}

	if (deviceNo >= numDevices)
		deviceNo = 0;

	// Erzeuge einen neuen Kontext
	context = clCreateContext(NULL, 1, devices + deviceNo, NULL, NULL, NULL);
	CHECK_SUCCESS("Error: creating OpenCL context")

	/*	NEUE WARTESCHLANGE
	*	Warteschlangen dienen dazu, Aufgaben an OpenCL-Gerate zu verteilen. Jede Aufgabe, wie
	*	das Ausfuhren eines Kernels oder ein Datentransfer zwischen Host-RAM und Device-RAM,
	*	muss in eine Warteschlange eingefugt werden. Dafur muss zunachst eine Warteschlange erstellt
	*	werden, die einem Gerat zugewiesen ist.
	*/
	commandQueue = clCreateCommandQueue(context, devices[deviceNo], CL_QUEUE_PROFILING_ENABLE, &status);
	CHECK_SUCCESS("Error: creating command queue")

	// Kerneldatei bestimmen, sie auslesen und in ein String speichern
	const char* filename = "Kernel.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	CHECK_SUCCESS("Error: loading OpenCL file")

	const char* source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	// Erstelle ein neues Programm aus dem Quelltext.
	program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
	CHECK_SUCCESS("Error: creating OpenCL program")

	// Kompiliere das Programm program fur dieses Gerat.
	status = clBuildProgram(program, 1, devices + deviceNo, NULL, NULL, NULL);
	if (status) {
		// Reserviere Speicher.
		char msg[120000];
		// Lese das Compiler-Log ein.
		clGetProgramBuildInfo(program, devices[deviceNo], CL_PROGRAM_BUILD_LOG, sizeof(msg), msg, NULL);
		cerr << "=== build failed ===\n" << msg << endl;
		getc(stdin);
		return FAILURE;
	}

	// Kernel aus dem Programm finden und extrahieren
	// Kernel für die Laufzeitumgebung bekanntgegeben
	praefixsumme256_kernel = clCreateKernel(program, "praefixsumme256_kernel", &status);
	CHECK_SUCCESS("Error: creating summe praefixsumme256_kernel")
	summe_kernel = clCreateKernel(program, "summe_kernel", &status);
	CHECK_SUCCESS("Error: creating summe summe_kernel")

	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}

	return status;
}

