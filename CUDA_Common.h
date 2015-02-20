#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
		cudaError _err = call;                                                    \
		if( cudaSuccess != _err) {                                                \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
			        __FILE__, __LINE__, cudaGetErrorString( _err) );              \
			exit(EXIT_FAILURE);                                                  \
		} } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
		CUDA_SAFE_CALL_NO_SYNC(call);                                           \
		cudaError _err = cudaThreadSynchronize();                                \
		if( cudaSuccess != _err) {                                               \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
			        __FILE__, __LINE__, cudaGetErrorString( _err) );              \
			exit(EXIT_FAILURE);                                                  \
		} } while (0)

#endif
