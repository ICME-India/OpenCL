//
//  main.c
//  send_data
//
//  Created by VISHAL SUBRAMANIAN on 3/6/18.
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
#include <OpenCL/OpenCL.h>
#define MAX_SOURCE_SIZE (0x100000)

int nx=5,ny=3,nz=5;
typedef struct {
    float ip;
    float im;
    float jp;
    float jm;
    float kp;
    float km;
    float ap;
    float con;
} coeffs;
int num = sizeof(coeffs)/sizeof(float);
float ***var1, *var2,*var_coeff;
coeffs ***cof;
cl_program program;
cl_kernel kernel_v1,kernel_v2,kernel_coeff;

cl_command_queue cmd_queue;
cl_context context;

cl_uint num_platforms;
cl_device_id device ;
cl_int error_stat;
size_t returned_size;
size_t buffer_size_var,buffer_size_coeff;

cl_mem device_coeff;
cl_mem device_var1,device_var2,device_var_coeff;

float ***
alloc3Dfloat (int nx, int ny, int nz)
{
    
    float *space;
    float ***arr3d;
    int i, j, k;
    
    /* first we set aside space for the array itself */
    
    space = (float *) malloc (nx * ny * nz * sizeof (float));
    
    /* next we allocate space of an array of pointers, each
     to eventually point to the first element of a
     2 dimensional array of pointers to pointers */
    
    arr3d = (float ***) malloc (nx * sizeof (float **));
    
    /* and for each of these we assign a pointer to a newly
     allocated array of pointers to a row */
    
    for (i = 0; i < nx; i++) {
        arr3d[i] = (float **) malloc (ny * sizeof (float *));
        
        /* and for each space in this array we put a pointer to
         the first element of each row in the array space
         originally allocated */
        
        for (j = 0; j < ny; j++) {
            arr3d[i][j] = space + (i * (ny * nz) + j * nz);
        }
    }
    
    /*  initialising all elements to 0.0 to be on safer side */
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                arr3d[i][j][k] = 0.0;
            }
        }
    }
    
    return (arr3d);
}
coeffs ***
alloc3Dcoeffs (int nx, int ny, int nz)
{
    
    coeffs *space;
    coeffs ***arr3d;
    int i, j, k;
    
    /* first we set aside space for the array itself */
    
    space = (coeffs *) malloc (nx * ny * nz * sizeof (coeffs));
    
    /* next we allocate space of an array of pointers, each
     to eventually point to the first element of a
     2 dimensional array of pointers to pointers */
    
    arr3d = (coeffs ***) malloc (nx * sizeof (coeffs **));
    
    /* and for each of these we assign a pointer to a newly
     allocated array of pointers to a row */
    
    for (i = 0; i < nx; i++) {
        arr3d[i] = (coeffs **) malloc (ny * sizeof (coeffs *));
        
        /* and for each space in this array we put a pointer to
         the first element of each row in the array space
         originally allocated */
        
        for (j = 0; j < ny; j++) {
            arr3d[i][j] = space + (i * (nz * ny) + j * nz);
        }
    }
    
    /*  initialising all elements to 0.0 to be on safer side */
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                arr3d[i][j][k].ip = 0.0;
                arr3d[i][j][k].jp = 0.0;
                arr3d[i][j][k].kp = 0.0;
                arr3d[i][j][k].im = 0.0;
                arr3d[i][j][k].jm = 0.0;
                arr3d[i][j][k].km = 0.0;
                arr3d[i][j][k].con = 0.0;
                arr3d[i][j][k].ap = 0.0;
            }
        }
    }
    
    return (arr3d);
}

int int_CL()
{
    printf("Initializing OpenCl constructs\n");
    /* load kernels */
    FILE *fp;
    const char fileName[] = "/Users/vishal_s/home/OpenCL/codes/send_data/send_data/send_data/kernel.cl";
    size_t source_size;
    char *source_str;
    
    /* Load kernel source file */
    fp = fopen(fileName,"rb");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel file.\n");
        exit(-1);
    }
    
    
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    /* Get the device information */
    error_stat = clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }
    /* Create a list of platform IDs */
    cl_platform_id platform[num_platforms];
    error_stat = clGetPlatformIDs(num_platforms, platform, NULL);
    if (error_stat!=CL_SUCCESS) {
        printf("Getting platforms %d \n", error_stat);
    }
    printf("\nNumber of OpenCL platforms: %d\n", num_platforms);
    printf("\n-------------------------\n");
    
    
    /* Investigate first platform */
    cl_char string[10240] = {0};
    /* Print out the platform name */
    error_stat = clGetPlatformInfo(platform[0], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
    printf("Platform: %s\n", string);
    
    /* Find the GPU CL device */
    error_stat = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    
    /* get data about the device */
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    error_stat = clGetDeviceInfo(device,CL_DEVICE_VENDOR,sizeof(vendor_name),vendor_name,&returned_size);
    error_stat = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    
    printf("Connecting to %s %s ....\n",vendor_name,device_name);
    
    /* set up context and command queue */
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_stat);
    cmd_queue    = clCreateCommandQueue(context, device, 0, NULL);
    
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &error_stat);
    
    if (error_stat!=CL_SUCCESS) {
        printf("program build with source error %d\n", error_stat);
    }
    error_stat = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error_stat!=CL_SUCCESS) {
        /* Determine the size of the log*/
        size_t log_size;
        printf("\nInside the if condition\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        /* Allocate memory for the log */
        char *log = (char *) malloc(log_size);
        
        /* Get the log */
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        
        /* Print the log */
        printf("%s\n", log);
    }
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error is clBuildProgram %d\n",error_stat);
        exit(-2);
    }
    
    kernel_v1 = clCreateKernel(program, "data_v1", &error_stat);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clCreateKernel %d\n",error_stat);
        exit(-3);
    }
    kernel_v2 = clCreateKernel(program, "data_v2", &error_stat);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clCreateKernel %d\n",error_stat);
        exit(-3);
    }
    kernel_coeff = clCreateKernel(program, "data_coeff", &error_stat);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clCreateKernel %d\n",error_stat);
        exit(-3);
    }
   
    /* Memory allocation in the gpu*/
    buffer_size_var   = sizeof(double) *nx*ny*nz;
    buffer_size_coeff  = sizeof(coeffs) *nx*ny*nz;
    device_var1   = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size_var  , NULL, NULL);
    device_var2   = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size_var  , NULL, NULL);
    device_var_coeff = clCreateBuffer(context, CL_MEM_READ_WRITE , buffer_size_coeff  , NULL, NULL);
    printf("Initializing OpenCl constructs..... Done\n");
    return 0;
}


void check_data()
{
    
    printf("Entering Check data\n");
    error_stat = clEnqueueWriteBuffer(cmd_queue,device_var1, CL_TRUE, 0,buffer_size_var,(void*)(&var1[0][0][0]),0, NULL, NULL);
    error_stat = clEnqueueWriteBuffer(cmd_queue,device_var2, CL_TRUE, 0,buffer_size_var,(void*)var2,0, NULL, NULL);
    error_stat = clEnqueueWriteBuffer(cmd_queue,device_var_coeff, CL_TRUE, 0,buffer_size_coeff,(void*)var_coeff,0, NULL, NULL);
    clFinish(cmd_queue);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueWriteBuffer %d\n",error_stat);
        exit(-4);
    }
    
    error_stat = clSetKernelArg(kernel_v1, 0, sizeof(cl_mem), &device_var1);
    error_stat = clSetKernelArg(kernel_v1, 1, sizeof(int), &nx);
    error_stat = clSetKernelArg(kernel_v1, 2, sizeof(int), &ny);
    error_stat = clSetKernelArg(kernel_v1, 3, sizeof(int), &nz);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clSetKernelArg %d\n",error_stat);
        exit(-5);
    }

    size_t global_work_size = 1;
    /*size_t work_size_pcr = 1;*/
    error_stat = clEnqueueNDRangeKernel(cmd_queue, kernel_v1, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueNDRangeKernel : %d\n",error_stat);
        exit(2);
    }
    clFinish(cmd_queue);
    error_stat = clSetKernelArg(kernel_v2, 0, sizeof(cl_mem), &device_var2);
    error_stat = clSetKernelArg(kernel_v2, 1, sizeof(int), &nx);
    error_stat = clSetKernelArg(kernel_v2, 2, sizeof(int), &ny);
    error_stat = clSetKernelArg(kernel_v2, 3, sizeof(int), &nz);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clSetKernelArg %d\n",error_stat);
        exit(-5);
    }
    /*size_t work_size_pcr = 1;*/
    error_stat = clEnqueueNDRangeKernel(cmd_queue, kernel_v2, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if(error_stat != CL_SUCCESS)
    {
        fprintf(stderr,"Error in clEnqueueNDRangeKernel : %d\n",error_stat);
        exit(2);
    }
//    error_stat = clSetKernelArg(kernel_coeff, 0, sizeof(cl_mem), &device_var_coeff);
//    error_stat = clSetKernelArg(kernel_coeff, 1, sizeof(int), &nx);
//    error_stat = clSetKernelArg(kernel_coeff, 2, sizeof(int), &ny);
//    error_stat = clSetKernelArg(kernel_coeff, 3, sizeof(int), &nz);
//    error_stat = clSetKernelArg(kernel_coeff, 4, sizeof(int), &num);
//    if(error_stat != CL_SUCCESS)
//    {
//      fprintf(stderr,"Error in clSetKernelArg %d\n",error_stat);
//       exit(-5);
//     }
//    /*size_t work_size_pcr = 1;*/
//    size_t global_work_size = 1;
//    error_stat = clEnqueueNDRangeKernel(cmd_queue, kernel_coeff, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
//    if(error_stat != CL_SUCCESS)
//     {
//        fprintf(stderr,"Error in clEnqueueNDRangeKernel : %d\n",error_stat);
//        exit(2);
//     }
//
    clFinish(cmd_queue);
    printf("Exiting Check data\n");
    return;
    
    
}

int del_CL()
{
    /*Release the memory*/
    clReleaseMemObject(device_var1);
    clReleaseMemObject(device_var2);
    clReleaseMemObject(device_var_coeff);
    clReleaseMemObject(device_coeff);
    return 0;
}


int main(int argc, const char * argv[]) {
    
    int i,j,k,count;
    int ip=0,im=1,jp=2,jm=3,kp=4,km=5,ap=6,con=7;
    int x8=nz*ny*num,y8=nz*num,z8=num;
    int x1=nz*ny,y1=nz,z1=1;
    var1 = alloc3Dfloat(nx,ny,nz);
    cof = alloc3Dcoeffs (nx,ny,nz);
    var2 =( float *) malloc (nx * ny * nz * sizeof (float));
    var_coeff = ( float *) malloc (nx * ny * nz * sizeof (coeffs));
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                var1[i][j][k] = k + j*nz + i*(ny*nz);
            }
        }
    }
    count = 0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                var1[i][j][k] = count+1;
                count++;
                var2[i*x1+j*y1+k*z1] =count;
//                cof[i][j][k].ip  = count+1;
//                cof[i][j][k].jp  = count+2;
//                cof[i][j][k].kp  = count+3;
//                cof[i][j][k].im  = count+4;
//                cof[i][j][k].jm  = count+5;
//                cof[i][j][k].km  = count+6;
//                cof[i][j][k].con = count+7;
//                cof[i][j][k].ap  = count+8;
//                count =count +8;
//                var_coeff[i*x8 + j*y8 + k*z8 + ip] = cof[i][j][k].ip;
//                var_coeff[i*x8 + j*y8 + k*z8 + im] = cof[i][j][k].im;
//                var_coeff[i*x8 + j*y8 + k*z8 + jp] = cof[i][j][k].jp;
//                var_coeff[i*x8 + j*y8 + k*z8 + jm] = cof[i][j][k].jm;
//                var_coeff[i*x8 + j*y8 + k*z8 + kp] = cof[i][j][k].kp;
//                var_coeff[i*x8 + j*y8 + k*z8 + km] = cof[i][j][k].km;
//                var_coeff[i*x8 + j*y8 + k*z8 + ap] = cof[i][j][k].ap;
//                var_coeff[i*x8 + j*y8 + k*z8 + con] = cof[i][j][k].con;
            }
        }
    }
//    for (i = 0; i < nx; i++) {
//        for (j = 0; j < ny; j++) {
//            for (k = 0; k < nz; k++) {
//
//                if(var_coeff[i*x8 + j*y8 + k*z8 + ip] != cof[i][j][k].ip)
//                    printf("Error in i=%d,j=%d,k=%d, ip\n",i,j,k);
//                if(var_coeff[i*x8 + j*y8 + k*z8 + im] != cof[i][j][k].im )
//                    printf("Error in i=%d,j=%d,k=%d, im\n",i,j,k);;
//                if(var_coeff[i*x8 + j*y8 + k*z8 + jp] != cof[i][j][k].jp )
//                    printf("Error in i=%d,j=%d,k=%d, jp\n",i,j,k);;
//                if(var_coeff[i*x8 + j*y8 + k*z8 + jm] != cof[i][j][k].jm )
//                    printf("Error in i=%d,j=%d,k=%d, jm\n",i,j,k);;
//                if(var_coeff[i*x8 + j*y8 + k*z8 + kp] != cof[i][j][k].kp )
//                    printf("Error in i=%d,j=%d,k=%d, kp\n",i,j,k);;
//                if(var_coeff[i*x8 + j*y8 + k*z8 + km] != cof[i][j][k].km )
//                    printf("Error in i=%d,j=%d,k=%d, km\n",i,j,k);;
//                if(var_coeff[i*x8 + j*y8 + k*z8 + ap] != cof[i][j][k].ap )
//                    printf("Error in i=%d,j=%d,k=%d, ap\n",i,j,k);;
//                if(var_coeff[i*x8 + j*y8 + k*z8 + con] != cof[i][j][k].con )
//                    printf("Error in i=%d,j=%d,k=%d, con\n",i,j,k);;
//
//            }
//        }
//    }
//    for (i=0;i<nx*ny*nz*num;i++)
//     var_coeff[i] = i;
//            for (i = 0; i < nx; i++) {
//                for (j = 0; j < ny; j++) {
//                    for (k = 0; k < nz; k++) {
//                        printf("i=%d,j=%d,k=%d,cof.ip=%f,var=%f\n",i,j,k,cof[i][j][k].ip,var_coeff[i*x8 + j*y8 + k*z8 + ip]);
//                        printf("i=%d,j=%d,k=%d,cof.im=%f,var=%f\n",i,j,k,cof[i][j][k].im,var_coeff[i*x8 + j*y8 + k*z8 + im]);
//                        printf("i=%d,j=%d,k=%d,cof.jp=%f,var=%f\n",i,j,k,cof[i][j][k].jp,var_coeff[i*x8 + j*y8 + k*z8 + jp]);
//                        printf("i=%d,j=%d,k=%d,cof.jm=%f,var=%f\n",i,j,k,cof[i][j][k].jm,var_coeff[i*x8 + j*y8 + k*z8 + jm]);
//                        printf("i=%d,j=%d,k=%d,cof.kp=%f,var=%f\n",i,j,k,cof[i][j][k].kp,var_coeff[i*x8 + j*y8 + k*z8 + kp]);
//                        printf("i=%d,j=%d,k=%d,cof.km=%f,var=%f\n",i,j,k,cof[i][j][k].km,var_coeff[i*x8 + j*y8 + k*z8 + km]);
//                        printf("i=%d,j=%d,k=%d,cof.ap=%f,var=%f\n",i,j,k,cof[i][j][k].ap,var_coeff[i*x8 + j*y8 + k*z8 + ap]);
//                        printf("i=%d,j=%d,k=%d,cof.con=%f,var=%f\n",i,j,k,cof[i][j][k].con,var_coeff[i*x8 + j*y8 + k*z8 + con]);
//                    }
//                }
//            }
    /*for (i=0;i<nx*ny*nz*num;i++)
        printf("%d=%f\n",i,var_coeff[i]);*/
    int_CL();
    check_data();
    del_CL();
    printf(" %lu %d \n",sizeof(coeffs),num);

    return 0;
}
