kernel void loopsmmult(global float *c) {
	unsigned int xid = get_global_id(0);
	float id = (float)xid;
	float loopsum = 0.0f;
	for (int y=0;y<72;y++) {
		for (int x=0;x<128;x++) {
			loopsum += (id+x)*y;
		}
	}
	c[xid] = loopsum;
}

kernel void loopsfill(global float *img) {
	unsigned int xid = get_global_id(0);
	img[xid*5+0] = 0.0f;
	img[xid*5+1] = 0.0f;
	img[xid*5+2] = 0.0f;
	img[xid*5+3] = 0.0f;
	img[xid*5+4] = INFINITY;
}
