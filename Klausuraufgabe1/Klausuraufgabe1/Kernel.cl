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
#define BLOCK_SIZE 256

__kernel void praefixsumme256_kernel(__global int* in, __global int* b_out, __global int* c_out, __global int* d_out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);		// wichtig für C

	__local int localArray[BLOCK_SIZE];
	
	int k = 8;	// depth of tree: log2(256)
	int d, i, i1, i2;

	// copy to local memory
	localArray[lid] = in[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-Sweep
	int noItemsThatWork = (BLOCK_SIZE/2);
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
	if (lid == (BLOCK_SIZE-1)) {
		localArray[(BLOCK_SIZE-1)] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	noItemsThatWork = 1;
	offset = (BLOCK_SIZE/2);
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

	// B output
	b_out[gid] = localArray[lid];
	
	// C output
	if (lid == (BLOCK_SIZE-1)) {
		int a_last_item = in[gid];
		int b_last_item = b_out[gid];
		int c_ergebnis = a_last_item+b_last_item;
		c_out[groupid] = c_ergebnis;
		//printf("%d. %d + %d = %d \n", groupid, a_last_item, b_last_item, c_ergebnis);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	
	// D output
	for (int i = 1; i <= BLOCK_SIZE; i++) {
		int c_last_item = c_out[i-1];
		int d_last_item = d_out[i-1];
		d_out[i] = c_last_item + d_last_item;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void summe_kernel(__global int* inB, __global int* inD, __global int* out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);		// wichtig für C

	__local int localArrayB[BLOCK_SIZE];
	__local int localArrayD[BLOCK_SIZE];
	
	// copy data to local array and add pairs of array elements
	localArrayB[lid] = inB[gid];
	localArrayD[lid] = inD[gid];

	barrier(CLK_LOCAL_MEM_FENCE);
	
	out[gid] = inB[gid] + inD[groupid];
}