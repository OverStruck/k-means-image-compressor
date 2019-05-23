#ifndef KMEANS_COMPRESSOR_H
#define KMEANS_COMPRESSOR_H

#include "dep/lodepng/lodepng.h"
#include<random>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <sstream>
#include "RGB_TRIPLET.h"

// global variables
//Number of clusters
extern int nCentroids;
// nCentroids * sizeof(int)
extern int PALETTE_BYTES;
// width * height * sizeof(int)
extern int IMAGE_BYTES;

// Number of threads
#define BLOCK_SIZE 16
#define GRID_SIZE 256

// nCentroids and size on device
extern __constant__ int dev_nCentroids;
extern __constant__ int dev_size;

// R,G,B Centroid's triple on device
//max number of centroid is 256
extern __constant__ int dev_RedCentroid[256];
extern __constant__ int dev_GreenCentroid[256];
extern __constant__ int dev_BlueCentroid[256];


void compress(RGB_TRIPLET *&inputImgRGB, RGB_TRIPLET *&outputImgRGB, int &inImgSize);
void initRGB_TRIPLET(RGB_TRIPLET *&rgbTriplet, int BYTES);
void initCentroids(const RGB_TRIPLET *inputImgRGB, RGB_TRIPLET *centroidsRGB, int nCentroids);

#endif//KMEANS_COMPRESSOR_H