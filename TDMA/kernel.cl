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
__kernel void ADD_ARR( __global float* result,__global float* result_odd,__global float* result_even)
{
    int i = get_global_id(0);
    if(i%2==0)
    {
        result[i] = result_even[i/2];
        
    }
    else
    {
        result[i] = result_odd[((i-1)/2)];
    }
    
}

__kernel void PCR_GPU(__global int* row_val,__global float* aw_gpu,__global float* ap_gpu,__global float*ae_gpu, __global float* su_gpu,__global float* aw_even,__global float* ap_even,__global float* ae_even, __global float* su_even,__global float* aw_odd,__global float* ap_odd,__global float* ae_odd, __global float* su_odd)
{
    int id,j,rows;
    id = get_global_id(0);
    rows = row_val[0];
    //printf("%d,%d\n",id,rows);
    //printf("%lf,%lf,%lf,%lf\n",aw_gpu[id],ap_gpu[id],ae_gpu[id],su_gpu[id]);
    if(id==0)
    {
        aw_even[0] = 0.0;
        ap_even[0] = ap_gpu[0] - aw_gpu[1]*ae_gpu[0]/ap_gpu[1];
        ae_even[0] = -ae_gpu[1]*ae_gpu[0]/ap_gpu[1];
        su_even[0] = su_gpu[0] - su_gpu[1]*ae_gpu[0]/ap_gpu[1];
        //printf("%d,%lf,%lf,%lf,%lf\n",id,aw_even[id],ap_even[id],ae_even[id],su_even[id]);
        //printf("%s\n","I am id 0");
    }
    else if(id==rows-1)
    {
        j = rows/2 -1;
        aw_odd[j] = -aw_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1];
        ap_odd[j] = ap_gpu[id] - ae_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1];
        ae_odd[j] = 0.0;
        su_odd[j] = su_gpu[id] - su_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1];
        //printf("%d,%lf,%lf,%lf,%lf\n",id,aw_odd[j],ap_odd[j],ae_odd[j],su_odd[j]);
        //printf("%s\n","I am id rows-1");
    }
    else if(id%2==0)
    {
        j = id/2;
        aw_even[j] = -aw_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1];
        ap_even[j] = ap_gpu[id] - ae_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1]- aw_gpu[id+1]*ae_gpu[id]/ap_gpu[id+1];
        ae_even[j] = -ae_gpu[id+1]*ae_gpu[id]/ap_gpu[id+1];
        su_even[j] = su_gpu[id] - su_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1]- su_gpu[id+1]*ae_gpu[id]/ap_gpu[id+1];
        //printf("%d,%lf,%lf,%lf,%lf\n",id,aw_even[j],ap_even[j],ae_even[j],su_even[j]);
        //printf("%s\n","I am id even");
    }
    else
    {
        j = (id-1)/2;
        aw_odd[j] = -aw_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1];
        ap_odd[j] = ap_gpu[id] - ae_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1]- aw_gpu[id+1]*ae_gpu[id]/ap_gpu[id+1];
        ae_odd[j] = -ae_gpu[id+1]*ae_gpu[id]/ap_gpu[id+1];
        su_odd[j] = su_gpu[id] - su_gpu[id-1]*aw_gpu[id]/ap_gpu[id-1]- su_gpu[id+1]*ae_gpu[id]/ap_gpu[id+1];
        //printf("%d,%lf,%lf,%lf,%lf\n",id,aw_odd[j],ap_odd[j],ae_odd[j],su_odd[j]);
        //printf("%s\n","I am id odd");
    }
    
    
}

__kernel void TDMA_GPU( __global int* row_val,__global float* aw,__global float* ap,__global float* ae, __global float* su,__global float *ans)
{
    int i=0;
    int n = row_val[0]/2;
    /*for (int i=0;i<(n);i++)
     {
     printf("%.6f , %.6f , %.6f , %.6f  \n",aw[i],ap[i],ae[i],su[i]);
     }
     printf("\n");*/
    // Forward Elimination
    ae[0] = ae[0]/ap[0];
    su[0] = su[0]/ap[0];
    ap[0] = 1.0;
    for(i=1;i<n;i++)
    {
        ap[i] = ap[i] - ae[i-1]*aw[i];
        su[i] = su[i] - su[i-1]*aw[i];
        aw[i] = 0.0;
        ae[i] = ae[i]/ap[i];
        su[i] = su[i]/ap[i];
        ap[i] = 1.0;
    }
    //Back Substitution
    ans[n-1] = su[n-1];
    for(i=n-2;i>=0;i--)
    {
        ans[i] = su[i] - ae[i]*ans[i+1];
    }
    /*for (int i=0;i<(n);i++)
    {
        printf("%.6f ",ans[i]);
    }
    printf("\n");*/
}

