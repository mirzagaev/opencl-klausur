/******************
Input array:
1 1 1 1 1 1 1 1		1 1 1 1 1 1 1 1		1 1 1 1 1 1 1 1

Upsweep result:
1 2 1 4 1 2 1 8		1 2 1 4 1 2 1 8		1 2 1 4 1 2 1 8

Downsweep (final) result is:
0 1 2 3 4 5 6 7		8 9 10 11 12 13	14 15	16 17 18 19 20 21 22 23

4 left-shifted by 1 = 8:		4 << 1;
4 right-shifted by 1 = 2:		4 >> 1;
******************/

// Thread block size
#define BLOCK_SIZE 8

__kernel void praefixsumme256_kernel(__global int* in, __global int* out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);
	
	__local int localArray[8];
	
	//printf("%d - %d \n", gid, lid);
	
	int k = 3;	// depth of tree: log2(8)
	int d, i, i1, i2;

	// copy to local memory
	localArray[lid] = in[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-Sweep
	int noItemsThatWork = 4;
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
	if (lid == 7) {
		localArray[7] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	noItemsThatWork = 1;
	offset = 4;
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
	
	if (lid == 7) {
		int a_last_item = in[gid];
		int b_last_item = localArray[7];
		int c_ergebnis = a_last_item+b_last_item;
		printf("%d + %d = %d \n", a_last_item, b_last_item, c_ergebnis);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// write result to global memory
	out[gid] = localArray[lid];
}