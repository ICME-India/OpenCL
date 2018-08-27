//
// Copyright (c) 2018, Vishal_S
// All rights reserved. Please read the "license.txt" for license terms.
//
// Project Title: OpenCL
//
// Developer: Vishal S
//
// Contact Info: vishalsubbu97@gmail.com
//
__kernel void
add_vec(__global float *a,
        __global float *b,
        __global float *ans)
{
    int gid = get_global_id(0);
    ans[gid] = a[gid]+b[gid];
}
