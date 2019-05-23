#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include "RGB_TRIPLET.h"
#include "kmeansCompressor.h"
#include "dep/ColorSpace/src/Comparison.h"

int nCentroids = 0;
// nCentroids * sizeof(int)
int PALETTE_BYTES = 0;
// width * height * sizeof(int)
int IMAGE_BYTES = 0;

void printColor(const ColorSpace::Rgb &color, int count)
{
	std::ostringstream colorBar;
	colorBar << "\x1b[38;2;" << color.r << ";" << color.g << ";" << color.b << "m" 
	<< "█████████████████" << "\x1b[0m\n";
		
	printf("Centroid[%d] = [%0.0lf, %0.0lf, %0.0lf]\t=>\t%s", count, color.r, color.g, color.b, colorBar.str().c_str());
}

void printColor(const RGB_TRIPLET *color, int i)
{
	int r = color->red[i];
	int g = color->green[i];
	int b = color->blue[i];
	std::ostringstream colorBar;
	colorBar << "\x1b[38;2;" << r << ";" << g << ";" << b << "m" 
	<< "█████████████████" << "\x1b[0m\n";
		
	printf("Centroid[%d] = [%d, %d, %d]\t=>\t%s", i, r, g, b, colorBar.str().c_str());
}

//checks file file exits
inline bool exits(const char* fileName)
{
  struct stat buffer;   
  return (stat (fileName, &buffer) == 0); 
}

void loadRawImage(const char* inputFile, RGB_TRIPLET *&inputImgRGB, int &inImgSize)
{
	printf("Loading image...\n");
	unsigned int width, height;
	std::vector<unsigned char> inImg;
	unsigned error = lodepng::decode(inImg, width, height, inputFile, LCT_RGB);
    
	if(error)
	{
		std::cout << "Decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		exit(-1);
	}
	
	IMAGE_BYTES = (width*height) * sizeof(int);
	initRGB_TRIPLET(inputImgRGB, IMAGE_BYTES);
	inputImgRGB->width = width;
	inputImgRGB->height = height;
	inImgSize = width * height;
	inputImgRGB->size = inImgSize;
	
	printf("W x H: %d x %d\n", inputImgRGB->width, inputImgRGB->height);
	
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

//pick random pixel from image
void getRandCentroid(const RGB_TRIPLET *inputImgRGB, std::vector<ColorSpace::Rgb> &colors, int i)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> xDis(0, inputImgRGB->width);
	std::uniform_int_distribution<> yDis(0, inputImgRGB->height);
	
	int x, y, r, g, b;
	x = xDis(gen);
	y = yDis(gen);
	r = inputImgRGB->red[x*y];
	g = inputImgRGB->green[x*y];
	b = inputImgRGB->blue[x*y];
	colors[i] = ColorSpace::Rgb(r,g,b);
}

//update centroid with specific rgb values
void updateCentroids(RGB_TRIPLET *centroidsRGB, const std::vector<ColorSpace::Rgb> &colors, int count)
{
	centroidsRGB->red[count] = colors[count].r;
	centroidsRGB->green[count] = colors[count].g;
	centroidsRGB->blue[count] = colors[count].b;
}

//initialize centroids
void initCentroids(const RGB_TRIPLET *inputImgRGB, RGB_TRIPLET *centroidsRGB, int nCentroids)
{
	printf("Initializing %d centroids at random:\n", nCentroids);
	double d;
	int count = 0, i = 0;
	std::vector<ColorSpace::Rgb> colors(nCentroids);
	getRandCentroid(inputImgRGB, colors, i);
	updateCentroids(centroidsRGB, colors, i);
	printColor(colors[i], i);
	++i;
	while (i < nCentroids)
	{
		getRandCentroid(inputImgRGB, colors, i);
		d = ColorSpace::Cie2000Comparison::Compare(&colors[0], &colors[i]);
		if (d > 20)
		{
			//printf("Distance between 0 and %d = %.2f\n", count, d);
			printColor(colors[i], i);
			//update GPU memory
			//updateCentroids(centroidsRGB, colors, i);
			++i;
		}
	}
	
	++count;
	bool canGo;
	//update centroid 1 before loop
	updateCentroids(centroidsRGB, colors, count);
	while(count < nCentroids)
	{
		for (i = count; i < nCentroids; ++i)
		{
			//printf("i=%d; i < %d\n", i, nCentroids);
			canGo = false;
			while ( canGo == false && (i < (nCentroids -1)) )
			{
				//printf("Comparing colors[%d] with colors[%d]\n", count, i+1);
				//printColor(colors[count], count);
				//printColor(colors[i+1], i+1);
				d = ColorSpace::Cie2000Comparison::Compare(&colors[count], &colors[i+1]);
				//printf("Distance: %.2f\n", d);
				if (d < 20)
				{
					//printf("FUCK d < 20\n");
					getRandCentroid(inputImgRGB, colors, i+1);
				}
				else
				{
					//printf("WE GOOOD. Updating....\n");
					updateCentroids(centroidsRGB, colors, i+1);
					canGo = true;
				}
			}
		}
		++count;
	}
	printf("\nUpdated Centroids:\n");
	for (i = 0; i < nCentroids; ++i)
		printColor(centroidsRGB, i);
	
}


int main(int argc, char** argv) {
	
	//check parameters
    if(argc != 4) 
	{
        printf("USAGE: %s <inputFileName.png> <outputFileName.png> <Number of Clusters(K)>\n", argv[0]);
        return -1;
    }
	
	//make sure input file exits
    const char* inputFile = argv[1];
	if (!exits(inputFile))
	{
		printf("ERROR: File '%s' does not exits.\n", inputFile);
        return -1;
	}
	
	// init device
	cudaSetDevice(0);
	CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	cudaDeviceSynchronize();
	
    const char* outputFile = argv[2];
	nCentroids = atoi(argv[3]);
	//only allow up to 256 clusters/colors
	nCentroids = (nCentroids > 256) ? 256 : nCentroids;
	
	//input & output image data
	int inImgSize;
	RGB_TRIPLET *inputImgRGB;
	RGB_TRIPLET *outputImgRGB;

	loadRawImage(inputFile, inputImgRGB, inImgSize);
    compress(inputImgRGB, outputImgRGB, inImgSize);
	saveRawImage(outputFile, outputImgRGB, inImgSize);

    delete outputImgRGB;
	
	std::cout << "Done" << std::endl;
	
    return 0;
}