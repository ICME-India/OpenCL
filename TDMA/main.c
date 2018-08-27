//
//  main.c
//  TDMA
//
//  Created by VISHAL SUBRAMANIAN on 9/3/18.
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
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "PCR_gpu.h"


#include <OpenCL/OpenCL.h>

#define MAX_SOURCE_SIZE (0x100000)
#define rows (512)


void TDMA(float *aw,float *ap,float *ae,float *su,float *result,int n)
{
    int i=0;
    //Forward Elimination
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
    result[n-1] = su[n-1];
    for(i=n-2;i>=0;i--)
    {
        result[i] = su[i] - ae[i]*result[i+1];
    }
    /*free(aw);
    free(ae);
    free(ap);
    free(su);*/
    
    
}
int PCR(float *ae,float *ap,float *aw,float *su,float *temp_pcr,int n)
{
    int i=0,j=0,k=0;
    float *ae_odd = (float *) malloc((n/2)*sizeof(float));
    float *ap_odd = (float *) malloc((n/2)*sizeof(float));
    float *aw_odd = (float *) malloc((n/2)*sizeof(float));
    float *su_odd = (float *) malloc((n/2)*sizeof(float));
    float *result_odd = (float *) malloc((n/2)*sizeof(float));
    float *ae_even = (float *) malloc((n/2)*sizeof(float));
    float *ap_even = (float *) malloc((n/2)*sizeof(float));
    float *aw_even = (float *) malloc((n/2)*sizeof(float));
    float *su_even = (float *) malloc((n/2)*sizeof(float));
    float *result_even = (float *) malloc((n/2)*sizeof(float));
    
    // even variable
    aw_even[0] = 0.0;
    ap_even[0] = ap[0] - aw[1]*ae[0]/ap[1];
    ae_even[0] = -ae[1]*ae[0]/ap[1];
    su_even[0] = su[0] - su[1]*ae[0]/ap[1];
    
    for(j=1;j<n/2;j++)
    {
        i = 2*j;
        aw_even[j] = -aw[i-1]*aw[i]/ap[i-1];
        ap_even[j] = ap[i] - ae[i-1]*aw[i]/ap[i-1] - aw[i+1]*ae[i]/ap[i+1];
        ae_even[j] = -ae[i+1]*ae[i]/ap[i+1];
        su_even[j] = su[i] - su[i-1]*aw[i]/ap[i-1] - su[i+1]*ae[i]/ap[i+1];
        
        k = 2*j-1;
        aw_odd[j-1] = -aw[k-1]*aw[k]/ap[k-1];
        ap_odd[j-1] = ap[k] - ae[k-1]*aw[k]/ap[k-1] - aw[k+1]*ae[k]/ap[k+1];
        ae_odd[j-1] = -ae[k+1]*ae[k]/ap[k+1];
        su_odd[j-1] = su[k] - su[k-1]*aw[k]/ap[k-1] - su[k+1]*ae[k]/ap[k+1];
    }
    j = (n/2)-1;
    aw_odd[j] = -aw[n-2]*aw[n-1]/ap[n-2];
    ap_odd[j] = ap[n-1] - ae[n-2]*aw[n-1]/ap[n-2];
    ae_odd[j] = 0.0;
    su_odd[j] = su[n-1] - su[n-2]*aw[n-1]/ap[n-2];
    /*free(ae);
    free(ap);
    free(aw);
    free(su);*/
    /*for (int i=0;i<(n/2);i++)
    {
        printf("%.6f , %.6f , %.6f , %.6f  \n",aw_even[i],ap_even[i],ae_even[i],su_even[i]);
    }
    printf("\n");*/
    if(n<1039)
    {
        TDMA(aw_odd,ap_odd,ae_odd,su_odd,result_odd,n/2);
        TDMA(aw_even,ap_even,ae_even,su_even,result_even,n/2);
    }
    else
    {
        PCR(ae_odd,ap_odd,aw_odd,su_odd,result_odd,n/2);
        PCR(ae_even,ap_even,aw_even,su_even,result_even,n/2);
    }
    
    
    for(j=0;j<n/2;j++)
    {
        temp_pcr[2*j]   = result_even[j];
        temp_pcr[2*j+1] = result_odd[j];
    }
    free(aw_even);
    free(ap_even);
    free(ae_even);
    free(su_even);
    free(result_even);
    free(aw_odd);
    free(ap_odd);
    free(ae_odd);
    free(su_odd);
    free(result_odd);
    return 0;
}

int main(int argc, const char * argv[]) {
    clock_t end;
    clock_t begin;
    int i=0;
    float time_spent ;
    float error = 0.0;
    
    //Domian size
    float delx = 2.0;
    float T0   = 10.0;
    float Tn   = 2000.0;
    //Allocate some memory and a place for the results
    float *ae = (float *) malloc(rows*sizeof(float));
    float *ap = (float *) malloc(rows*sizeof(float));
    float *aw = (float *) malloc(rows*sizeof(float));
    float *su = (float *) malloc(rows*sizeof(float));
    float *temp_cpu = (float *) malloc(rows*sizeof(float));
    float *temp_gpu = (float *) malloc(rows*sizeof(float));
    
    //fill in the values
    for(i=0;i<rows;i++)
    {
        ap[i] = -2.0/(delx*delx);
        ae[i] = 1.0/(delx*delx);
        aw[i] = 1.0/(delx*delx);
        su[i] = 0.0;
    }

    su[0]      = su[0] - aw[0]*T0;
    su[rows-1] = su[rows-1] - ae[rows-1]*Tn;
    ae[rows-1] = 0.0;
    aw[0]      = 0.0;
    
    //do the calculation in cpu - serial execution
    begin = clock();
    TDMA(aw,ap,ae,su,temp_cpu,rows);
    //PCR(ae,ap,aw,su,temp_cpu,rows);
    end = clock();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    printf ("Time taken for execution in cpu:%lf \n",time_spent);
    //Do the OpenCL calculation
    //Allocate some memory and a place for the results
    /*ae = (float *) malloc(rows*sizeof(float));
    ap = (float *) malloc(rows*sizeof(float));
    aw = (float *) malloc(rows*sizeof(float));
    su = (float *) malloc(rows*sizeof(float));*/
    //fill in the values
    for(i=0;i<rows;i++)
    {
        ap[i] = -2.0/(delx*delx);
        ae[i] = 1.0/(delx*delx);
        aw[i] = 1.0/(delx*delx);
        su[i] = 0.0;
    }
    su[0]      = su[0] - aw[0]*T0;
    su[rows-1] = su[rows-1] - ae[rows-1]*Tn;
    ae[rows-1] = 0.0;
    aw[0]      = 0.0;
    int_CL(rows);
    begin = clock();
    //TDMACL(ae,ap,aw,su,temp_pcr,rows);
    PCR_CL(ae,ap,aw,su,temp_gpu,rows);
    end = clock();
    del_CL();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    printf ("Time taken for execution of PCR:%lf\n",time_spent);
    //check the result
    for(i=0;i<rows;i=i+2)
    {
        error += (temp_gpu[i]-temp_cpu[i])*(temp_gpu[i]-temp_cpu[i]);
    }
    
    error = sqrt(error)/rows;
    if (error < 0.001)
        printf("Calculation succesful\n");
    else
        printf("Calculation failed\n");
    
    //Print result
    /*for (i=0;i<rows;i++)
    {
        printf("%.6f  ",temp_cpu[i]);
    }
    printf("\n");
    for (i=0;i<rows;i++)
    {
        printf("%.6f  ",temp_gpu[i]);
    }
    printf("\n");*/

    //free up memory
    free(ae);
    free(ap);
    free(aw);
    free(su);
    free(temp_cpu);
    free(temp_gpu);
    return 0;
}
