#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

#ifndef RGB_TRIPLET_H
#define RGB_TRIPLET_H
class RGB_TRIPLET
{
	public:
		int *red, *green, *blue;
		unsigned int width, height, size;
		CUDA_CALLABLE_MEMBER RGB_TRIPLET(int IMAGE_BYTES, int w, int h)
		{
			red = new int[IMAGE_BYTES];
			green = new int[IMAGE_BYTES];
			blue = new int[IMAGE_BYTES];
			
			width = w;
			height = h;
			size = w * h;
		}
		
		CUDA_CALLABLE_MEMBER ~RGB_TRIPLET()
		{
			delete[] red;
			delete[] green;
			delete[] blue;
		}
};

#endif //RGB_TRIPLET_H