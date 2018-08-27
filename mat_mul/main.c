//
//  main.c
//  mat_mul
//
//  Created by VISHAL SUBRAMANIAN on 14/2/18.
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
#include <time.h>
#include <math.h>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include <OpenCL/OpenCL.h>

#define MAX_SOURCE_SIZE (0x100000)
#define rows (8)
#define cols (8)
//"   const int row,                                                 \n" \

const char *KernelSource = "\n" \
"__kernel void mat_mul(                                            \n" \
"   __global int* row_val,                                                 \n" \
"   __global float* a,                                             \n" \
"   __global float* b,                                             \n" \
"   __global float *ans)                                           \n" \
"{                                                                 \n" \
"   int i = get_global_id(0);                                      \n" \
"   int row = row_val[0];                                                \n" \
"   for(int j=0;j<row;j++)                                         \n" \
"       ans[i] += a[(i/row)*row + j] * b[j*row + i%row];                  \n" \
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

int runCL(float *a,float *b,float *result, int m,int n)
{
    cl_program program[1];
    cl_kernel kernel[1];
    
    cl_command_queue cmd_queue;
    cl_context context;
    
    cl_device_id device = NULL;
    cl_int err = 0;
    size_t returned_size = 0;
    size_t buffer_size;
    
    cl_mem a_mem,b_mem,ans_mem,row_cl;
    
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
    kernel[0] = clCreateKernel(program[0], "mat_mul", &err);
    
    // Memory allocation in the gpu
    
    
    buffer_size = sizeof(float) *m*n;
    //input size of row
    row_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, row_cl, CL_TRUE, 0, sizeof(int),&m,0, NULL, NULL);
    // input a
    a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, a_mem, CL_TRUE, 0, buffer_size,(void*)a,0, NULL, NULL);
    //input b
    b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b_mem, CL_TRUE, 0, buffer_size,(void*)b,0, NULL, NULL);
    
    //output
    ans_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, ans_mem, CL_TRUE, 0, buffer_size,(void*)result,0, NULL, NULL);
    /*printf("Result to gpu\n");
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            printf("%0.2f ",result[i*n + j]);
        }
        printf("\n");
    }*/
    //Wait for this to happen
    clFinish(cmd_queue);
    
    // Kernel arguments
    
    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &row_cl);
    if(err != CL_SUCCESS)
    {
        fprintf(stderr,"Error is here\n");
        exit(1);
    }
    err = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &a_mem);
    err = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &b_mem);
    err = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &ans_mem);
    
    
    //execution
    
    size_t global_work_size = m*n;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clFinish(cmd_queue);
    
    // read the data
    err = clEnqueueReadBuffer(cmd_queue,ans_mem,CL_TRUE,0,buffer_size,result,0 ,NULL,NULL);
    /*printf("Result from gpu\n");
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            printf("%0.2f ",result[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");*/
    //RElease the mempry
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(ans_mem);
    
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
    return 0;
}
int main(int argc, const char * argv[]) {
    clock_t end;
    clock_t begin;
    float time_spent ;
    float error = 0.0;
    //Allocate some memory and a place for the results
    float *a = (float *) malloc(rows*cols*sizeof(float));
    float *b = (float *) malloc(rows*cols*sizeof(float));
    float *result_cpu = (float *) malloc(rows*cols*sizeof(float));
    float *result_gpu = (float *) malloc(rows*cols*sizeof(float));
    
    //fill in the values
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
        {
            a[i*cols + j] = rand()%10;
            b[i*cols + j] = rand()%10;
            result_cpu[i*cols + j] = 0.0;
            result_gpu[i*cols + j] = 0.0;
        }
    /*printf("Values of a\n");
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            printf("%0.2f ",a[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("Values of b\n");
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            printf("%0.2f ",b[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");*/
    /*printf("Initial Values of result\n");
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            printf("%0.2f ",result_cpu[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");*/
    //do the calculation in cpu - serial execution
    begin = clock();
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            for(int k=0;k<cols;k++)
            {
                result_cpu[i*cols + j] += a[i*cols + k] * b[k*cols + j];
            }
        }
    }
    /*printf("Final Values of result\n");
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            printf("%0.2f ",result_cpu[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");*/
    end = clock();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    printf ("Time taken for execution in cpu:%lf\n",time_spent);
    //Do the OpenCL calculation
    begin = clock();
    runCL(a,b,result_gpu,rows,cols);
    end = clock();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    printf ("Time taken for execution in gpu:%lf\n",time_spent);
    //check the result
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            
            error += (result_cpu[i*cols + j]-result_gpu[i*cols + j])*(result_cpu[i*cols + j]-result_gpu[i*cols + j]);

        }
    }
    // Print result from cpu
    /*for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            printf("%0.2f ",result_cpu[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    // Print result from gpu
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            printf("%0.2f ",result_gpu[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");*/

    error = sqrt(error);
    if (error < 0.00001)
        printf("Calculation succesful\n");
    else
        printf("Calculation failed\n");
    //free up memory
    free(a);
    free(b);
    free(result_cpu);
    free(result_gpu);
    
    
    return 0;
}

