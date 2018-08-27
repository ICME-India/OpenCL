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
__kernel void data_v1(__global float* device_var1,  const int nx,  const int ny, const int nz )
{
    int i,j,k;
    int x1=ny*nz ,y1=nz ,z1=1;
    float val ;
    for ( i=0;i<nx;i++)
        for (j=0;j<ny;j++)
            for(k=0;k<nz;k++)
            {
                val =device_var1[i*x1 + j*y1 + k*z1];
                printf("i=%d,j=%d,k=%d, val = %lf\n",i,j,k,val);
            }
    
}

__kernel void data_v2(__global float* device_var2,  const int nx,  const int ny, const int nz )
{
    int i,j,k;
    int x1=ny*nz ,y1=nz ,z1=1;
    float val ;
    for ( i=0;i<nx;i++)
        for (j=0;j<ny;j++)
            for(k=0;k<nz;k++)
            {
                val =device_var2[i*x1 + j*y1 + k*z1];
                printf("i=%d,j=%d,k=%d, val = %lf\n",i,j,k,val);
            }
    
}
__kernel void data_coeff(__global float* device_var_coeff,  const int nx,  const int ny, const int nz,const int num )
{
    int i,j,k;
    int ip=0,im=1,jp=2,jm=3,kp=4,km=5,ap=6,con=7;
    int x8=nz*ny*num,y8=nz*num,z8=num;
    float val ;
                for (i = 0; i < nx; i++) {
                    for (j = 0; j < ny; j++) {
                        for (k = 0; k < nz; k++) {
                            printf("i=%d,j=%d,k=%d,cof.ip=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + ip]);
                            printf("i=%d,j=%d,k=%d,cof.im=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + im]);
                            printf("i=%d,j=%d,k=%d,cof.jp=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + jp]);
                            printf("i=%d,j=%d,k=%d,cof.jm=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + jm]);
                            printf("i=%d,j=%d,k=%d,cof.kp=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + kp]);
                            printf("i=%d,j=%d,k=%d,cof.km=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + km]);
                            printf("i=%d,j=%d,k=%d,cof.ap=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + ap]);
                            printf("i=%d,j=%d,k=%d,cof.con=%f\n",i,j,k,device_var_coeff[i*x8 + j*y8 + k*z8 + con]);
                        }
                    }
                }
    
}
