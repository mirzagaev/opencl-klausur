/*
Zentraler Bestandteil jedes OpenCL-Programms ist der Kernel, der von der GPU ausgefuhrt
wird. In diesem Abschnitt werden die Grundlagen der Kernel-Programmierung in der Programmiersprache
OpenCL erlautert.
Die Programmiersprache OpenCL ist von C abgeleitet. Ein OpenCL-Kernel hat daher Ahnlichkeit
mit einer C-Funktion. Es werden jedoch keine Daten zuruckgegeben, daher handelt
es sich um eine Funktion mit Ruckgabetyp void, wie an folgendem Beispiel deutlich wird:
*/

__kernel void praefixsumme256_kernel(__global int* in, __global int* out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);
	
	__local int localArray[256];
	
	int k = 8;	// depth of tree: log2(256)
	int d, i, i1, i2;

	// copy to local memory
	localArray[lid] = in[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-Sweep
	int noItemsThatWork = 128;
	int offset = 1;
	for (d = 0; d<k; d++, noItemsThatWork >>= 1, offset <<= 1) {
		if (lid < noItemsThatWork) {
			i1 = lid*(offset << 1) + offset - 1;
			i2 = i1 + offset;
			localArray[i2] = localArray[i1] + localArray[i2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Down-Sweep
	if (lid == 255)
		localArray[255] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	noItemsThatWork = 1;
	offset = 128;
	for (d = 0; d<k; d++, noItemsThatWork <<= 1, offset >>= 1)
	{
		if (lid < noItemsThatWork) {
			i1 = lid*(offset << 1) + offset - 1;
			i2 = i1 + offset;
			int tmp = localArray[i1];
			localArray[i1] = localArray[i2];
			localArray[i2] = tmp + localArray[i2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result to global memory
	out[gid] = localArray[lid];
}
/*
In diesem Fall kann die Kernel-Funktion Calculate vom Host-Code aus aufgerufen werden,
die Funktion Sum dagegen nur innerhalb des Device-Codes.
*/