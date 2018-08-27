//
//  PCR_gpu.h
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
#ifndef PCR_gpu_h
#define PCR_gpu_h

#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


#include <OpenCL/OpenCL.h>





char * load_program_source(const char *filename);
int int_CL(int n);
int del_CL(void);
int PCR_CL(float *ae,float *ap,float *aw,float *su,float *temp_gpu,int n);



#endif /* PCR_gpu_h */
