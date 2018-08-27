//
//  PCR_gpu.c
//  TDMA
//
//  Created by VISHAL SUBRAMANIAN on 10/3/18.
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
#include "PCR_gpu.h"

// variables required for OpenCl implementation
cl_program program;
cl_kernel kernel_pcr, kernel_tdma, kernel_add;

cl_command_queue cmd_queue, tdma_queue_1, tdma_queue_2;
cl_context context;

cl_device_id device ;
cl_int error_stat;
size_t returned_size;
size_t buffer_size;

cl_mem ae_gpu,ap_gpu,aw_gpu,su_gpu,results_gpu;
cl_mem ae_odd,ap_odd,aw_odd,su_odd,results_odd;
cl_mem ae_even,ap_even,aw_even,su_even,results_even;
cl_mem row_cl;

#define MAX_SOURCE_SIZE (0x100000)
/*__kernel void PCR_GPU(__global int* row_val,__global float* aw,__global float* ap,__global float* ae,__global float* su,__global float* aw_even,__global float* ap_even,__global float* ae_even, __global float* su_even,__global float* aw_odd,__global float* ap_odd,__global float* ae_odd,__global float* su_odd)                                        \n" \
"{                                                                  \n" \
    "    int i=0,j=0,k=0;                                                \n" \
    "   int rows = row_val[0];                                          \n" \
    "   aw_even[j] = 0.0; \n" \
    "   ap_even[j] = ap[j] - aw[j+1]*ae[j]/ap[j+1]; \n" \
    "   ae_even[j] = -ae[j+1]*ae[j]/ap[j+1]; \n" \
    "   su_even[j] = su[j] - su[1]*ae[j]/ap[j+1]; \n" \
    "   for(j=1;j<rows/2;j++) \n" \
        "   { \n" \
            "       i = 2*j; \n" \
            "       aw_even[j] = -aw[i-1]*aw[i]/ap[i-1]; \n" \
            "       ap_even[j] = ap[i] - ae[i-1]*aw[i]/ap[i-1] - aw[i+1]*ae[i]/ap[i+1]; \n" \
            "       ae_even[j] = -ae[i+1]*ae[i]/ap[i+1]; \n" \
            "       su_even[j] = su[i] - su[i-1]*aw[i]/ap[i-1] - su[i+1]*ae[i]/ap[i+1]; \n" \
            "       k = 2*j-1; \n" \
            "       aw_odd[j-1] = -aw[k-1]*aw[k]/ap[k-1]; \n" \
            "       ap_odd[j-1] = ap[k] - ae[k-1]*aw[k]/ap[k-1] - aw[k+1]*ae[k]/ap[k+1]; \n" \
            "       ae_odd[j-1] = -ae[k+1]*ae[k]/ap[k+1]; \n" \
            "       su_odd[j-1] = su[k] - su[k-1]*aw[k]/ap[k-1] - su[k+1]*ae[k]/ap[k+1]; \n" \
            "   } \n" \
    "   j = (rows/2)-1; \n" \
    "   aw_odd[j] = -aw[rows-2]*aw[rows-1]/ap[rows-2]; \n" \
    "   ap_odd[j] = ap[rows-1] - ae[rows-2]*aw[rows-1]/ap[rows-2]; \n" \
    "   ae_odd[j] = 0.0; \n" \
    "   su_odd[j] = su[rows-1] - su[rows-2]*aw[rows-1]/ap[rows-2];                                        \n" \
    "}                                                                  \n" \
"\n";*/


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
int int_CL(int n)
{
    // load kernels
    FILE *fp;
    const char fileName[] = "/Users/vishal_s/home/OpenCL/codes/TDMA/TDMA/kernel.cl";
    size_t source_size;
    char *source_str;
    
    /* Load kernel source file */
    fp = fopen(fileName,"rb");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel file.\n");
        exit(10);
    }
    
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    // Get the device information
    
    //Find the GPU CL device
    error_stat = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    
    //get data about the device
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    error_stat = clGetDeviceInfo(device,CL_DEVICE_VENDOR,sizeof(vendor_name),vendor_name,&returned_size);
    error_stat = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    
    printf("Connecting to %s %s ....\n",vendor_name,device_name);
    
    //set up context and command queue
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_stat);
    cmd_queue    = clCreateCommandQueue(context, device, 0, NULL);
    tdma_queue_1 = clCreateCommandQueue(context, device, 0, NULL);
    tdma_queue_2 = clCreateCommandQueue(context, device, 0, NULL);
    
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &error_stat);
    if (error_stat!=CL_SUCCESS) {
        printf("program build with source error %d\n", error_stat);
    }
    error_stat = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error is clBuildProgram %d\n",error_stat);
        exit(11);
    }
    kernel_tdma = clCreateKernel(program, "TDMA_GPU", &error_stat);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clCreateKernel %d\n",error_stat);
        exit(12);
    }
    kernel_pcr = clCreateKernel(program, "PCR_GPU", &error_stat);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clCreateKernel %d\n",error_stat);
        exit(12);
    }
    kernel_add = clCreateKernel(program, "ADD_ARR", &error_stat);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clCreateKernel %d\n",error_stat);
        exit(12);
    }
    // Memory allocation in the gpu
    buffer_size = sizeof(float) *n;
    //input size of row
    row_cl = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int), NULL, NULL);
    ae_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size, NULL, NULL);
    ap_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size, NULL, NULL);
    aw_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size, NULL, NULL);
    su_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size, NULL, NULL);
    // allocate memory for even and odd
    ae_even = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    ap_even = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    aw_even = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    su_even = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    
    
    ae_odd = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    ap_odd = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    aw_odd = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    su_odd = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    
    //output
    results_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    results_even = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    results_odd = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size/2, NULL, NULL);
    return 0;
}

int del_CL()
{
    //RElease the mempry
    clReleaseMemObject(ae_gpu);
    clReleaseMemObject(ap_gpu);
    clReleaseMemObject(aw_gpu);
    clReleaseMemObject(su_gpu);
    clReleaseMemObject(results_gpu);
    clReleaseMemObject(ae_odd);
    clReleaseMemObject(ap_odd);
    clReleaseMemObject(aw_odd);
    clReleaseMemObject(su_odd);
    clReleaseMemObject(results_odd);
    clReleaseMemObject(ae_even);
    clReleaseMemObject(ap_even);
    clReleaseMemObject(aw_even);
    clReleaseMemObject(su_even);
    clReleaseMemObject(results_even);
    clReleaseMemObject(row_cl);
    
    clReleaseCommandQueue(cmd_queue);
    clReleaseCommandQueue(tdma_queue_1);
    clReleaseCommandQueue(tdma_queue_2);
    clReleaseContext(context);
    
    return 0;
}
int PCR_CL(float *ae,float *ap,float *aw,float *su,float *temp_gpu,int n)
{
    
    error_stat = clEnqueueWriteBuffer(cmd_queue, row_cl, CL_TRUE, 0, sizeof(int),&n,0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueWriteBuffer n %d\n",error_stat);
        exit(13);
    }
    error_stat = clEnqueueWriteBuffer(cmd_queue, ae_gpu, CL_TRUE, 0, buffer_size,(void*)ae,0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueWriteBuffer ae %d\n",error_stat);
        exit(13);
    }
    error_stat = clEnqueueWriteBuffer(cmd_queue, ap_gpu, CL_TRUE, 0, buffer_size,(void*)ap,0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueWriteBuffer ap %d\n",error_stat);
        exit(13);
    }
    error_stat = clEnqueueWriteBuffer(cmd_queue, aw_gpu, CL_TRUE, 0, buffer_size,(void*)aw,0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueWriteBuffer aw %d\n",error_stat);
        exit(13);
    }
    error_stat = clEnqueueWriteBuffer(cmd_queue, su_gpu, CL_TRUE, 0, buffer_size,(void*)su,0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueWriteBuffer su %d\n",error_stat);
        exit(13);
    }
    //Wait for this to happen
    clFinish(cmd_queue);

    // Kernel arguments
    
    error_stat = clSetKernelArg(kernel_pcr, 0, sizeof(cl_mem), &row_cl);
   
    error_stat = clSetKernelArg(kernel_pcr, 1, sizeof(cl_mem), &aw_gpu);
    error_stat = clSetKernelArg(kernel_pcr, 2, sizeof(cl_mem), &ap_gpu);
    error_stat = clSetKernelArg(kernel_pcr, 3, sizeof(cl_mem), &ae_gpu);
    error_stat = clSetKernelArg(kernel_pcr, 4, sizeof(cl_mem), &su_gpu);
    
    error_stat = clSetKernelArg(kernel_pcr, 5, sizeof(cl_mem), &aw_even);
    error_stat = clSetKernelArg(kernel_pcr, 6, sizeof(cl_mem), &ap_even);
    error_stat = clSetKernelArg(kernel_pcr, 7, sizeof(cl_mem), &ae_even);
    error_stat = clSetKernelArg(kernel_pcr, 8, sizeof(cl_mem), &su_even);
    
    error_stat = clSetKernelArg(kernel_pcr,  9, sizeof(cl_mem), &aw_odd);
    error_stat = clSetKernelArg(kernel_pcr, 10, sizeof(cl_mem), &ap_odd);
    error_stat = clSetKernelArg(kernel_pcr, 11, sizeof(cl_mem), &ae_odd);
    error_stat = clSetKernelArg(kernel_pcr, 12, sizeof(cl_mem), &su_odd);
    
   
    
    size_t global_work_size = n;
    //size_t work_size_pcr = 1;
    error_stat = clEnqueueNDRangeKernel(cmd_queue, kernel_pcr, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error is here : %d\n",error_stat);
        exit(2);
    }
    clFinish(cmd_queue);
    
    error_stat = clSetKernelArg(kernel_tdma, 0, sizeof(cl_mem), &row_cl);
    error_stat = clSetKernelArg(kernel_tdma, 1, sizeof(cl_mem), &aw_even);
    error_stat = clSetKernelArg(kernel_tdma, 2, sizeof(cl_mem), &ap_even);
    error_stat = clSetKernelArg(kernel_tdma, 3, sizeof(cl_mem), &ae_even);
    error_stat = clSetKernelArg(kernel_tdma, 4, sizeof(cl_mem), &su_even);
    error_stat = clSetKernelArg(kernel_tdma, 5, sizeof(cl_mem), &results_even);
    
    size_t global_work_size_tdma = 1;
    size_t work_size_tdma = 1;
    error_stat = clEnqueueNDRangeKernel(tdma_queue_1, kernel_tdma, 1, NULL, &global_work_size_tdma, &work_size_tdma, 0, NULL, NULL);
    
    error_stat = clSetKernelArg(kernel_tdma, 0, sizeof(cl_mem), &row_cl);
    error_stat = clSetKernelArg(kernel_tdma, 1, sizeof(cl_mem), &aw_odd);
    error_stat = clSetKernelArg(kernel_tdma, 2, sizeof(cl_mem), &ap_odd);
    error_stat = clSetKernelArg(kernel_tdma, 3, sizeof(cl_mem), &ae_odd);
    error_stat = clSetKernelArg(kernel_tdma, 4, sizeof(cl_mem), &su_odd);
    error_stat = clSetKernelArg(kernel_tdma, 5, sizeof(cl_mem), &results_odd);
    error_stat = clEnqueueNDRangeKernel(tdma_queue_2, kernel_tdma, 1, NULL, &global_work_size_tdma, &work_size_tdma, 0, NULL, NULL);
    clFinish(tdma_queue_2);
    clFinish(tdma_queue_1);
    //float *temp_odd = (float *) malloc((n/2)*sizeof(float));

    error_stat = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), &results_gpu);
    error_stat = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), &results_odd);
    error_stat = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), &results_even);
    error_stat = clEnqueueNDRangeKernel(cmd_queue, kernel_add, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clFinish(cmd_queue);
    // read the data
    error_stat = clEnqueueReadBuffer(cmd_queue,results_gpu,CL_TRUE,0,buffer_size,temp_gpu,0 ,NULL,NULL);
    clFinish(cmd_queue);
    
    return 0;
}
