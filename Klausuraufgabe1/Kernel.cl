// Thread block size
#define BLOCK_SIZE 8

__kernel void praefixsumme256_kernel(__global int* in, __global int* out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);
	int size_x = get_local_size(0);
	
	__local int localArray[8];
	
	// copy to local memory
	localArray[lid] = in[gid];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//printf("localArray[get_global_id(0)]: %d; \n",localArray[gid<<1]);

	// loop for each block 0....7, 8....15, ....
	for(int n=0; n<(size_x/BLOCK_SIZE); n++) {
		if (lid < BLOCK_SIZE) {
			int x = (n*BLOCK_SIZE)+lid;
			printf("%d:: %d \n", n, localArray[x]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//out[gid] = localArray[lid];
}

__kernel void praefixsumme256_kerne2(__global int* in, __global int* out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);
	
	__local int localArray[8];
	
	int k = 3;	// depth of tree: log2(256)
	int d, i, i1, i2;

	// printf("gid %d; \n",gid);
	// printf("\nlid %d; \n",lid);

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
		// printf("\nlocalArray=%d\n", i2);
		// printf("%d \n", localArray[i2]);
	}

	// Down-Sweep
	if (lid == 7)
		localArray[7] = 0;
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
	
	out[gid] = localArray[lid];
}

__kernel void praefixsumme256_kernel3(__global int* in, __global int* out)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupid = get_group_id(0);

	printf("gid: %d \n", gid);
	printf("lid: %d \n", lid);
	printf("groupid: %d \n", groupid);
	
	__local int localArray[8];

	int k = 3;	// depth of tree: log2(256)
	int d, i, i1, i2;

	// copy to local memory
	localArray[lid] = in[gid];
	// localArray[2*lid] = in[lid]; // load input into shared memory temp[2*lid+1] = g_idata[2*lid+1]; 
	barrier(CLK_LOCAL_MEM_FENCE);

	// Up-Sweep
	int noItemsThatWork = 4;
	int n = 4;
	int offset = 1;

	//	4 << 1;     // 4 left-shifted by 1 = 8
	//	4 >> 1;     // 4 right-shifted by 1 = 2

	for (d = n>>1; d > 0; d >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < d) {
			int i1 = offset*(2*lid+1)-1;
			int i2 = offset*(2*lid+2)-1;
			localArray[i2] += localArray[i1];
		}
		offset *= 2;
	}

	// Down-Sweep
	if (lid == 0)
		localArray[n-1] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (d = 1; d < n; d *= 2) {
		// traverse down tree & build scan
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < d){
			int i1 = offset*(2*lid+1)-1;
			int i2 = offset*(2*lid+2)-1;
			float t = localArray[i1];
			localArray[i1] = localArray[i2];
			localArray[i2] += t;
		}
	}
	
	// write result to global memory
	out[gid] = localArray[lid];
}