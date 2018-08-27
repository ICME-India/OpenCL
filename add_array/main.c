//
//  main.c
//  array_add
//
//  Created by VISHAL SUBRAMANIAN on 13/2/18.
//  Copyright © 2018 VISHAL SUBRAMANIAN. All rights reserved.
//

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
#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>

#include <OpenCL/OpenCL.h>

#define MAX_SOURCE_SIZE (0x100000)


const char *KernelSource = "\n" \
"__kernel void add_vec(                                            \n" \
"   __global float* a,                                             \n" \
"   __global float* b,                                             \n" \
"   __global float *ans)                                           \n" \
"{                                                                 \n" \
"   int i = get_global_id(0);                                      \n" \
"   ans[i] = a[i] + b[i];                                          \n" \
"}                                                                 \n" \
"\n";

char * load_program_source(const char *filename)
{
    
    struct stat statbuf;
    FILE *fh;
    char *source;
    
    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;
    
    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';
    
    return source;
}

int runCL(float *a,float *b,float *result, int n)
{
    cl_program program[1];
    cl_kernel kernel[1];
    
    cl_command_queue cmd_queue;
    cl_context context;
    
    cl_device_id device = NULL;
    cl_int err = 0;
    size_t returned_size = 0;
    size_t buffer_size;
    
    cl_mem a_mem,b_mem,ans_mem;
    
    // Get the device information
    
    //Find the GPU CL device
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    //get data about the device
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    err = clGetDeviceInfo(device,CL_DEVICE_VENDOR,sizeof(vendor_name),vendor_name,&returned_size);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    
    printf("Connecting to %s %s ....\n",vendor_name,device_name);
    
    //set up context and command queue
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    cmd_queue = clCreateCommandQueue(context, device, 0, NULL);
    
    // Program and kernel
    
    char *program_source ;
    /*FILE *fp;
    size_t source_size;
    
    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    program_source = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( program_source, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    */
    //program[0]= clCreateProgramWithSource(context, 1, (const char**)&program_source, (const size_t *)&source_size, &err);
    program[0]= clCreateProgramWithSource(context, 1, (const char**)&KernelSource,NULL, &err);
    err = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
    kernel[0] = clCreateKernel(program[0], "add_vec", &err);
    
    // Memory allocation in the gpu
    
    
    buffer_size = sizeof(float) *n;
    
    // input a
    a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, a_mem, CL_TRUE, 0, buffer_size,(void*)a,0, NULL, NULL);
    //input b
    b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b_mem, CL_TRUE, 0, buffer_size,(void*)b,0, NULL, NULL);
    
    //output
    ans_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    
    //Wait for this to happen
    clFinish(cmd_queue);
    
    // Kernel arguments
    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &a_mem);
    err = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &b_mem);
    err = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &ans_mem);
    
    //execution
    
    size_t global_work_size = n;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clFinish(cmd_queue);
    
    // read the data
    err = clEnqueueReadBuffer(cmd_queue,ans_mem,CL_TRUE,0,buffer_size,result,0 ,NULL,NULL);
    
    //RElease the mempry
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(ans_mem);
    
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
    return 0;
}
int main(int argc, const char * argv[]) {
    // Problem size
    int n=1000;
    
    //Allocate some memory and a place for the results
    float *a = (float *) malloc(n*sizeof(float));
    float *b = (float *) malloc(n*sizeof(float));
    float *result = (float *) malloc(n*sizeof(float));
    
    //fill in the values
    for(int i=0;i<n;i++)
    {
        a[i] = (float)i;
        b[i] = (float)(n-i);
        result[i] = 0.f;
    }
    
    //Do the OpenCL calculation
    runCL(a,b,result,n);
    
    for(int i=0;i<n;i++) printf("%f\n",result[i]);
    
    //free up memory
    free(a);
    free(b);
    free(result);
    return 0;
}
