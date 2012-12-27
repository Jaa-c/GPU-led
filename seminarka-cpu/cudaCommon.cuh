//----------------------------------------------------------------------------------------
/**
 * @file       cudaCommon.cuhh
 * @author     Daniel Princ
 * @date       2012/12/13
 *
 *  Common methods a macros used in cuda.
 *
*/
//----------------------------------------------------------------------------------------


#ifndef __CUDACOMMON_CUH__
#define __CUDACOMMON_CUH__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "Voxel.h"

inline void HandleError( cudaError_t error, const char *file, int line )
{
	if (error != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
		//exit( EXIT_FAILURE );
	}
}

inline void __cudaCheckError( const char *file, const int line ) {
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
	}

	err = cudaDeviceSynchronize();
	if( cudaSuccess != err ) {
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
	}
}
#define CHECK_LAST_ERR() __cudaCheckError( __FILE__, __LINE__ )
#define CHECK_ERR( error ) HandleError( error, __FILE__, __LINE__ )


#endif