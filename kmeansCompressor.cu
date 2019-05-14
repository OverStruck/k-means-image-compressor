#include "dep/lodepng.h" //LodePNG for C (ISO C90) and C++
#include <iostream>
#include <stdio.h> //for printf inside Kernels
#include <cuda.h>
#include <cuda_runtime.h>
#include<random>
#include <sys/stat.h>

//Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}
  
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// Number of threads
#define BLOCK_SIZE 16
#define GRID_SIZE 256

// nCentroids and size on device
__constant__ int dev_nCentroids;
__constant__ int dev_size;

// global variables
int nCentroids = 0;
int PALETTE_BYTES = 0; // nCentroids * sizeof(int)
int IMAGE_BYTES = 0;  // width * height * sizeof(int)

// R,G,B Centroid's triple on device
//max number of centroid is 256
__constant__ int dev_RedCentroid[256];
__constant__ int dev_GreenCentroid[256];
__constant__ int dev_BlueCentroid[256];

inline bool exits(const char* inputFile)
{
  struct stat buffer;   
  return (stat (inputFile, &buffer) == 0); 
}

// Clears arrays before each kernel getClusterLabel iteration
__global__
void clearPaletteArrays(int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue, 
int* dev_pixelClusterCounter, int* dev_tempRedCentroid, int* dev_tempGreenCentroid, int* dev_tempBlueCentroid )
{
	// 1 block, 16x16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;
	//printf("clearPaletteArrays -> %d\n", threadID);
	if(threadID < dev_nCentroids)
	{
		// nCentroids long
		dev_sumRed[threadID] = 0;
		dev_sumGreen[threadID] = 0;
		dev_sumBlue[threadID] = 0;
		dev_pixelClusterCounter[threadID] = 0;
		dev_tempRedCentroid[threadID] = 0;
		dev_tempGreenCentroid[threadID] = 0;
		dev_tempBlueCentroid[threadID] = 0;
	}
}

// Clear label array before each kernel getClusterLabel iteration
__global__
void clearLabelArray(int *dev_labelArray)
{
	// Global thread index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int threadID = x + y * blockDim.x * gridDim.x;

	// labelArray is "size" long
	if(threadID < dev_size)
	{
		dev_labelArray[threadID] = 0;
	}
}

/*
 * Finds the minimum distance between each triple dev_Red[i] dev_Green[i] dev_Blue[i] and all centroids.
 * Then saves the equivalent centroid label in dev_labelArray.
 * labelArray is   "width*height" long, monodimensional array
 *
 * INPUT : pixel triple arrays dev_Red, dev_Green, dev_Blue. labelArray that will contains the label for each pixel triple
 */
__global__
void getClusterLabel(int *dev_Red, int *dev_Green, int *dev_Blue, int *dev_labelArray)
{

	// Global thread index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int threadID = x + y * blockDim.x * gridDim.x;

	//default min value of distance
	float min = 500.0, value;
	//will be label
	int index = 0;


	if(threadID < dev_size)
	{
		// Finding the nearest centroid to current triple identified by threadID thread
		for(int i = 0; i < dev_nCentroids; i++)
		{
			// Performing Euclidean distance, Saving current value
			value = sqrtf(powf((dev_Red[threadID]-dev_RedCentroid[i]),2.0) 
			+ 
			powf((dev_Green[threadID]-dev_GreenCentroid[i]),2.0) 
			+ 
			powf((dev_Blue[threadID]-dev_BlueCentroid[i]),2.0));

			if(value < min)
			{
				// saving new nearest centroid
				min = value;
				// Updating his index
				index = i;
			}
		}
		// Writing to global memory the index of the nearest centroid
		// for dev_Red[threadID], dev_Green[threadID], dev_Blue[threadID] pixel triple
		dev_labelArray[threadID] = index;
	}
}

// Summing Red, Green, Blue values per cluster

__global__
void sumCluster(int *dev_Red, int *dev_Green, int *dev_Blue, int *dev_sumRed,
int *dev_sumGreen, int *dev_sumBlue, int *dev_labelArray, int *dev_pixelClusterCounter)
{
	// Global thread index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int threadID = x + y * blockDim.x * gridDim.x;

	if(threadID < dev_size)
	{
		int currentLabelArray = dev_labelArray[threadID];
		int currentRed = dev_Red[threadID];
		int currentGreen = dev_Green[threadID];
		int currentBlue = dev_Blue[threadID];
		
		// Writing to global memory needs a serialization. Many threads are writing into the same few locations
		atomicAdd(&dev_sumRed[currentLabelArray], currentRed);
		atomicAdd(&dev_sumGreen[currentLabelArray], currentGreen);
		atomicAdd(&dev_sumBlue[currentLabelArray], currentBlue);
		atomicAdd(&dev_pixelClusterCounter[currentLabelArray], 1);
	}
}

/*
 *  Calculates the new R,G,B values of the centroids dividing 
 the sum of color (for each channel) by the number of pixels in that cluster
 *	New values are stored in global memory since the current R,G,B values of the centroids are in read-only constant memory.
 */
__global__
void newCentroids(int *dev_tempRedCentroid, int *dev_tempGreenCentroid, int *dev_tempBlueCentroid,
	int *dev_sumRed, int *dev_sumGreen, int *dev_sumBlue, int *dev_pixelClusterCounter)
{
	// 1 block , 16*16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids)
	{
		int currentPixelCounter = dev_pixelClusterCounter[threadID];
		int sumRed = dev_sumRed[threadID];
		int sumGreen = dev_sumGreen[threadID];
		int sumBlue = dev_sumBlue[threadID];
		
		//new RGB Centroids' values written in global memory
		dev_tempRedCentroid[threadID] = static_cast<int>( (sumRed/currentPixelCounter) );
		dev_tempGreenCentroid[threadID] = static_cast<int>( (sumGreen/currentPixelCounter) );
		dev_tempBlueCentroid[threadID] = static_cast<int>( (sumBlue/currentPixelCounter) );
	}

}

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

void initCentroids(const RGB_TRIPLET *inputImgRGB, RGB_TRIPLET *&centroidsRGB, int nCentroids)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> xDis(0, inputImgRGB->width);
	std::uniform_int_distribution<> yDis(0, inputImgRGB->height);
	
	int x, y;
	printf("Initializing %d centroids at random:\n", nCentroids);
	for (int i = 0; i < nCentroids; ++i)
	{
		x = xDis(gen);
		y = yDis(gen);
		centroidsRGB->red[i] = inputImgRGB->red[x*y];
		centroidsRGB->green[i] = inputImgRGB->green[x*y];
		centroidsRGB->blue[i] = inputImgRGB->blue[x*y];
		printf("Centroid[%d] = [%d, %d, %d]\n", i, centroidsRGB->red[i] , centroidsRGB->green[i], centroidsRGB->blue[i]);
	}
}

void loadRawImage(const char* inputFile, RGB_TRIPLET *&inputImgRGB, int &inImgSize)
{
	printf("Loading image...\n");
	std::vector<unsigned char> inImg;
	unsigned int width, height;
	unsigned error = lodepng::decode(inImg, width, height, inputFile, LCT_RGB);
    
	if(error)
	{
		std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		exit(-1);
	}
	
	IMAGE_BYTES = (width*height) * sizeof(int);
	inputImgRGB = new RGB_TRIPLET(IMAGE_BYTES, width, height);
	inImgSize = width * height;
	
	std::cout << "W x H: " << inputImgRGB->width << " x " << inputImgRGB->height << std::endl;
	
	printf("Preparing image...\n");
	int where = 0;
	for(int i = 0; i < inImgSize*3; i+=3)
	{
		inputImgRGB->red[where] = inImg.at(i);
		inputImgRGB->green[where] = inImg.at(i+1);
		inputImgRGB->blue[where] = inImg.at(i+2);
		++where;
	}
	printf("RGB channels loaded\n");
}

void saveRawImage(const char* outputFile, RGB_TRIPLET *&outputImgRGB,  int &inImgSize)
{
	// Prepare data for output
	printf("Writing compressed image...\n");
    std::vector<unsigned char> outImg;
	for( int i = 0; i < outputImgRGB->size; ++i)
	{
		outImg.push_back(outputImgRGB->red[i]);
		outImg.push_back(outputImgRGB->green[i]);
		outImg.push_back(outputImgRGB->blue[i]);
	}
	
	// Output the data
    unsigned error = lodepng::encode(outputFile, outImg, outputImgRGB->width, outputImgRGB->height, LCT_RGB);
	
	//if there's an error, display it
    if(error)
	{
		std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		exit(-1);
	}
}

void compress(RGB_TRIPLET *&inputImgRGB, RGB_TRIPLET *&outputImgRGB, int &inImgSize)
{
	IMAGE_BYTES = (inputImgRGB->size) * sizeof( int);
	PALETTE_BYTES = nCentroids * sizeof(int);
	
	RGB_TRIPLET *centroidsRGB = new RGB_TRIPLET(PALETTE_BYTES, nCentroids, 1);
	RGB_TRIPLET *sumRGB = new RGB_TRIPLET(PALETTE_BYTES, nCentroids, 1);
	
	// array containing ref to GPU label array variable
	 int *labelArray, *dev_labelArray;
	labelArray = new  int[IMAGE_BYTES];
	
	initCentroids(inputImgRGB, centroidsRGB, nCentroids);
	//loadPalette("image2_palette4.raw", centroidsRGB, nCentroids);
	
	// allocate memory on GPU
	// reference to GPU  Pixels'RGB values, Centroids' RGB values
	int *dev_Red, *dev_Green, *dev_Blue;
	int *dev_tempRedCentroid, *dev_tempGreenCentroid, *dev_tempBlueCentroid;
	int *dev_sumRed, *dev_sumGreen, *dev_sumBlue;
	// references to array where pixels' count are stored
	int *pixelClusterCounter, *dev_pixelClusterCounter;
	pixelClusterCounter = new  int[PALETTE_BYTES];
	
	CUDA_CALL(cudaMalloc((void**) &dev_Red, IMAGE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_Green, IMAGE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_Blue, IMAGE_BYTES));
	
	CUDA_CALL(cudaMalloc((void**) &dev_tempRedCentroid, PALETTE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_tempGreenCentroid, PALETTE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_tempBlueCentroid, PALETTE_BYTES));
	
	CUDA_CALL(cudaMalloc((void**) &dev_labelArray, IMAGE_BYTES));
	
	CUDA_CALL(cudaMalloc((void**) &dev_sumRed, PALETTE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_sumGreen, PALETTE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_sumBlue, PALETTE_BYTES));
	CUDA_CALL(cudaMalloc((void**) &dev_pixelClusterCounter, PALETTE_BYTES));

	// copy host CPU memory to GPU
	CUDA_CALL(cudaMemcpy(dev_Red, inputImgRGB->red, IMAGE_BYTES, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_Green, inputImgRGB->green, IMAGE_BYTES, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_Blue, inputImgRGB->blue, IMAGE_BYTES, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMemcpy(dev_tempRedCentroid, centroidsRGB->red, PALETTE_BYTES, cudaMemcpyHostToDevice ));
	CUDA_CALL(cudaMemcpy(dev_tempGreenCentroid, centroidsRGB->green, PALETTE_BYTES, cudaMemcpyHostToDevice ));
	CUDA_CALL(cudaMemcpy(dev_tempBlueCentroid, centroidsRGB->blue, PALETTE_BYTES, cudaMemcpyHostToDevice ));
	
	CUDA_CALL(cudaMemcpy(dev_labelArray, labelArray, IMAGE_BYTES, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMemcpy(dev_sumRed, sumRGB->red, PALETTE_BYTES, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_sumGreen, sumRGB->green, PALETTE_BYTES, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_sumBlue, sumRGB->blue, PALETTE_BYTES, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMemcpy(dev_pixelClusterCounter, pixelClusterCounter, PALETTE_BYTES, cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, centroidsRGB->red, PALETTE_BYTES));
	CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, centroidsRGB->green, PALETTE_BYTES));
	CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, centroidsRGB->blue, PALETTE_BYTES));
	CUDA_CALL(cudaMemcpyToSymbol(dev_nCentroids, &nCentroids, sizeof(int) ));
	CUDA_CALL(cudaMemcpyToSymbol(dev_size, &inputImgRGB->size, sizeof(int)));
	
	// Clearing centroids on host
	for(int i = 0; i < nCentroids; i++)
	{
		centroidsRGB->red[i] = 0;
		centroidsRGB->green[i] = 0;
		centroidsRGB->blue[i] = 0;
	}
	
	int BLOCK_X, BLOCK_Y;
	BLOCK_X = ceil(inputImgRGB->width/BLOCK_SIZE);
	BLOCK_Y = ceil(inputImgRGB->height/BLOCK_SIZE);
	
	if(BLOCK_X > GRID_SIZE)
		BLOCK_X = GRID_SIZE;
	if(BLOCK_Y > GRID_SIZE)
		BLOCK_Y = GRID_SIZE;
	
	//2D Grid
	//Minimum number of threads that can handle width¡height pixels
	dim3 dimGRID(BLOCK_X, BLOCK_Y);
	//2D Block
	//Each dimension is fixed
	dim3 dimBLOCK(BLOCK_SIZE, BLOCK_SIZE);
	
	printf("\nLaunching %d CUDA kernels\n\n", nCentroids);
	
	for(int i = 0; i < 5; i++)
	{
		// Passing image RGB components, palette RGB components, label Array, number of Clusters
		// Init  arrays' values to 0
		// Kernel needs only 1 block since nClusters
		clearPaletteArrays<<<1, dimBLOCK>>>(dev_sumRed, dev_sumGreen, dev_sumBlue, 
		dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid, dev_pixelClusterCounter);
		
		// Init labelarray values to 0
		clearLabelArray<<<dimGRID, dimBLOCK>>>(dev_labelArray);
		// Calculates the distance from each pixel and all centroids
		// Then saves the equivalent label in dev_labelArray
		getClusterLabel<<< dimGRID, dimBLOCK >>>(dev_Red, dev_Green, dev_Blue, dev_labelArray);
		//Sums RGB values in each Cluster
		sumCluster<<<dimGRID, dimBLOCK>>>(dev_Red, dev_Green, dev_Blue, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_labelArray, dev_pixelClusterCounter);
		//Finds new RGB Centroids' values
		newCentroids<<<1,dimBLOCK >>>(dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter);
		//Old RGB Centroids' values are in constant memory
		//Updated RGB Centroids' values are in global memory
		//We need a swap
		CUDA_CALL(cudaMemcpy(centroidsRGB->red, dev_tempRedCentroid, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(centroidsRGB->green, dev_tempGreenCentroid, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(centroidsRGB->blue, dev_tempBlueCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));
		//Uploading in constant memory updated RGB Centroids' values
		CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, centroidsRGB->red, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, centroidsRGB->green, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, centroidsRGB->blue, PALETTE_BYTES));	

	}
	//debug
	CUDA_CALL(cudaMemcpy(labelArray, dev_labelArray, IMAGE_BYTES, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(sumRGB->red, dev_sumRed, PALETTE_BYTES, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(sumRGB->green, dev_sumGreen, PALETTE_BYTES, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(sumRGB->blue, dev_sumBlue, PALETTE_BYTES, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(pixelClusterCounter, dev_pixelClusterCounter, PALETTE_BYTES, cudaMemcpyDeviceToHost));
	
	printf("New centroids:\n");
	for(int i = 0; i < nCentroids; i++) {
		printf("Centroid[%d] = [%d, %d, %d]\n", i, centroidsRGB->red[i], centroidsRGB->green[i], centroidsRGB->blue[i]);
	}
	
	printf("\nPreparing data for output...\n");
	outputImgRGB = new RGB_TRIPLET(IMAGE_BYTES, inputImgRGB->width, inputImgRGB->height);
	for( int i = 0; i < inputImgRGB->size; ++i)
	{
		outputImgRGB->red[i] = centroidsRGB->red[labelArray[i]];
		outputImgRGB->green[i] = centroidsRGB->green[labelArray[i]];
		outputImgRGB->blue[i] =centroidsRGB->blue[labelArray[i]];
	}
	
	delete[] labelArray;
	delete[] pixelClusterCounter;
	
	CUDA_CALL(cudaFree(dev_Red));
	CUDA_CALL(cudaFree(dev_Green));
	CUDA_CALL(cudaFree(dev_Blue));
	CUDA_CALL(cudaFree(dev_tempRedCentroid));
	CUDA_CALL(cudaFree(dev_tempGreenCentroid));
	CUDA_CALL(cudaFree(dev_tempBlueCentroid));
	CUDA_CALL(cudaFree(dev_labelArray));
	CUDA_CALL(cudaFree(dev_sumRed));
	CUDA_CALL(cudaFree(dev_sumGreen));
	CUDA_CALL(cudaFree(dev_sumBlue));
	CUDA_CALL(cudaFree(dev_pixelClusterCounter));
}

int main(int argc, char** argv) {
	//check parameters
    if(argc != 4) 
	{
        printf("USAGE: %s <inputFileName.png> <outputFileName.png> <Number of Clusters(K)>\n", argv[0]);
        return -1;
    }
	// Read the arguments
    const char* inputFile = argv[1];
	if (!exits(inputFile))
	{
		printf("ERROR: File '%s' does not exits.\n", inputFile);
        return -1;
	}
	
	// init device
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();
	
    const char* outputFile = argv[2];
	nCentroids = atoi(argv[3]);
	nCentroids = (nCentroids > 256) ? 256 : nCentroids;
	
	//input & output image data
	int inImgSize;
	RGB_TRIPLET *inputImgRGB;
	RGB_TRIPLET *outputImgRGB;

	loadRawImage(inputFile, inputImgRGB, inImgSize);
    compress(inputImgRGB, outputImgRGB, inImgSize);
	saveRawImage(outputFile, outputImgRGB, inImgSize);

    delete inputImgRGB;
    delete outputImgRGB;
	
	std::cout << "Done" << std::endl;
	
    return 0;
}





