#include "kmeansCompressor.h"

// nCentroids and size on device
__constant__ int dev_nCentroids;
__constant__ int dev_size;

// R,G,B Centroid's triple on device
//max number of centroid is 256
__constant__ int dev_RedCentroid[256];
__constant__ int dev_GreenCentroid[256];
__constant__ int dev_BlueCentroid[256];

void initRGB_TRIPLET(RGB_TRIPLET *&rgbTriplet, int BYTES)
{
	CUDA_CALL(cudaHostAlloc((void**) &rgbTriplet, sizeof(RGB_TRIPLET), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	CUDA_CALL(cudaHostAlloc((void**) &rgbTriplet->red, BYTES, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	CUDA_CALL(cudaHostAlloc((void**) &rgbTriplet->green, BYTES, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	CUDA_CALL(cudaHostAlloc((void**) &rgbTriplet->blue, BYTES, cudaHostAllocWriteCombined | cudaHostAllocMapped));
}

// Clears arrays before each kernel getClusterLabel iteration
void clearPaletteArrays(RGB_TRIPLET *rgbSums, RGB_TRIPLET *centroids, int *pixelClusterCounter)
{
	// 1 block, 16x16 threads
	//int threadID = threadIdx.x + threadIdx.y * blockDim.x;
	
	// Clearing centroids on host
	for(int i = 0; i < nCentroids; i++)
	{
		centroids->red[i] = 0;
		centroids->green[i] = 0;
		centroids->blue[i] = 0;
		rgbSums->red[i] = 0;
		rgbSums->green[i] = 0;
		rgbSums->blue[i] = 0;
		pixelClusterCounter[i] = 0;
	}
}

// Clear label array before each kernel getClusterLabel iteration
__global__
void clearLabelArray(int *labelArray)
{
	// Global thread index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int threadID = x + y * blockDim.x * gridDim.x;

	// labelArray is "size" long
	if(threadID < dev_size)
	{
		labelArray[threadID] = 0;
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
void getClusterLabel(RGB_TRIPLET *inputImgRGB, int *dev_labelArray)
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
			value = sqrtf(powf((inputImgRGB->red[threadID] - dev_RedCentroid[i]), 2.0) 
			+ 
			powf((inputImgRGB->green[threadID] - dev_GreenCentroid[i]), 2.0) 
			+ 
			powf((inputImgRGB->blue[threadID] - dev_BlueCentroid[i]), 2.0));

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
void sumCluster(const RGB_TRIPLET *inputImgRGB, RGB_TRIPLET *rgbSums, const int *labelArray, int *pixelClusterCounter)
{
	// Global thread index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int threadID = x + y * blockDim.x * gridDim.x;

	if(threadID < dev_size)
	{
		int currentLabelArray = labelArray[threadID];
		int currentRed = inputImgRGB->red[threadID];
		int currentGreen = inputImgRGB->green[threadID];
		int currentBlue = inputImgRGB->blue[threadID];
		
		// Writing to global memory needs a serialization. Many threads are writing into the same few locations
		atomicAdd(&rgbSums->red[currentLabelArray], currentRed);
		atomicAdd(&rgbSums->green[currentLabelArray], currentGreen);
		atomicAdd(&rgbSums->blue[currentLabelArray], currentBlue);
		atomicAdd(&pixelClusterCounter[currentLabelArray], 1);
	}
}

/*
 *  Calculates the new R,G,B values of the centroids dividing 
 the sum of color (for each channel) by the number of pixels in that cluster
 *	New values are stored in global memory since the current R,G,B values of the centroids are in read-only constant memory.
 */
__global__
void newCentroids(RGB_TRIPLET *rgbCentroids, const RGB_TRIPLET *rgbSums, int *pixelClusterCounter)
{
	// 1 block , 16*16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;
	
	if(threadID < dev_nCentroids)
	{
		int currentPixelCounter = pixelClusterCounter[threadID];
		int sumRed = rgbSums->red[threadID];
		int sumGreen = rgbSums->green[threadID];
		int sumBlue = rgbSums->blue[threadID];

		//new RGB Centroids' values written in global memory
		rgbCentroids->red[threadID] = static_cast<int>( (sumRed/currentPixelCounter) );
		rgbCentroids->green[threadID] = static_cast<int>( (sumGreen/currentPixelCounter) );
		rgbCentroids->blue[threadID] = static_cast<int>( (sumBlue/currentPixelCounter) );
	}
}

void compress(RGB_TRIPLET *&inputImgRGB, RGB_TRIPLET *&outputImgRGB, int &inImgSize)
{
	PALETTE_BYTES = nCentroids * sizeof(int);
	
	RGB_TRIPLET *rgbSums, *rgbCentroids;
	
	initRGB_TRIPLET(rgbSums, PALETTE_BYTES);
	initRGB_TRIPLET(rgbCentroids, PALETTE_BYTES);
	
	rgbSums->width = nCentroids;
	rgbSums->height = 1;
	
	rgbCentroids->width = nCentroids;
	rgbCentroids->height = 1;
	
	int *labelArray, *pixelClusterCounter;
	CUDA_CALL(cudaHostAlloc((void**) &labelArray, IMAGE_BYTES, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	CUDA_CALL(cudaHostAlloc((void**) &pixelClusterCounter, PALETTE_BYTES, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	
	initCentroids(inputImgRGB, rgbCentroids, nCentroids);

	CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, rgbCentroids->red, PALETTE_BYTES));
	CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, rgbCentroids->green, PALETTE_BYTES));
	CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, rgbCentroids->blue, PALETTE_BYTES));
	CUDA_CALL(cudaMemcpyToSymbol(dev_nCentroids, &nCentroids, sizeof(int) ));
	CUDA_CALL(cudaMemcpyToSymbol(dev_size, &inImgSize, sizeof(int) ));
	
	// Clearing centroids on host
	for(int i = 0; i < nCentroids; i++)
	{
		rgbCentroids->red[i] = 0;
		rgbCentroids->green[i] = 0;
		rgbCentroids->blue[i] = 0;
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
	
	printf("\nLaunching CUDA kernels\n");
	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n\n",
	dimGRID.x, dimGRID.y, dimGRID.z, dimBLOCK.x, dimBLOCK.y, dimBLOCK.z);
	
	for(int i = 0; i < 1; i++)
	{
		// Init  arrays' values to 0
		// Kernel needs only 1 block since nClusters
		clearPaletteArrays(rgbSums, rgbCentroids, pixelClusterCounter);
		// Init labelarray values to 0
		clearLabelArray<<<dimGRID, dimBLOCK>>>(labelArray);
		cudaDeviceSynchronize();
		// Calculates the distance from each pixel and all centroids
		// Then saves the equivalent label in dev_labelArray
		getClusterLabel<<< dimGRID, dimBLOCK >>>(inputImgRGB, labelArray);
		cudaDeviceSynchronize();
		//Sums RGB values in each Cluster
		sumCluster<<<dimGRID, dimBLOCK>>>(inputImgRGB, rgbSums, labelArray, pixelClusterCounter);
		cudaDeviceSynchronize();
		//Finds new RGB Centroids' values
		newCentroids<<<1, dimBLOCK >>>(rgbCentroids, rgbSums, pixelClusterCounter);
		cudaDeviceSynchronize();
		//Uploading in constant memory updated RGB Centroids' values
		CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, rgbCentroids->red, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, rgbCentroids->green, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, rgbCentroids->blue, PALETTE_BYTES));	
		cudaDeviceSynchronize();
	}
	
	//debug info
	int r, g, b;
	std::ostringstream colorBar;
	int nCentroidsP = nCentroids > 16 ? 16 : nCentroids;
	printf("New centroids:\n");
	for(int i = 0; i < nCentroidsP; ++i)
	{
		r = rgbCentroids->red[i];
		g = rgbCentroids->green[i];
		b = rgbCentroids->blue[i];

		colorBar << "\x1b[38;2;" << r << ";" << g << ";" << b << "m" << "█████████████████" << "\x1b[0m\n";

		printf("Centroid[%d] = [%d, %d, %d]\t=>\t%s", i, r, g, b, colorBar.str().c_str());
		colorBar.str("");
	}
	
	printf("\nPreparing data for output...\n");
	outputImgRGB = new RGB_TRIPLET(IMAGE_BYTES, inputImgRGB->width, inputImgRGB->height);
	for( int i = 0; i < inputImgRGB->size; ++i)
	{
		outputImgRGB->red[i] = rgbCentroids->red[labelArray[i]];
		outputImgRGB->green[i] = rgbCentroids->green[labelArray[i]];
		outputImgRGB->blue[i] =rgbCentroids->blue[labelArray[i]];
	}
	
	CUDA_CALL(cudaFreeHost(rgbSums));
	CUDA_CALL(cudaFreeHost(labelArray));
	CUDA_CALL(cudaFreeHost(inputImgRGB));
	CUDA_CALL(cudaFreeHost(rgbCentroids));
	CUDA_CALL(cudaFreeHost(pixelClusterCounter));
}



