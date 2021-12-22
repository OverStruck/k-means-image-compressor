#include "exceptions.h"
#include "InputParser.h"
#include "ImageCompressor.h"
#include "HelperFunctions.h"
#include "Config.h"

#include <iostream>

int main(int argc, char **argv)
{
	kmic::printWelcome();
	try
	{
		kmic::InputParser input(argc, argv);
		kmic::Config config(input);

		Centroids centroids(config.getK());
		Image *inputImage = kmic::loadRawImage(config.getIfn());
		kmic::initCentroids(centroids, inputImage);

		kmic::ImageCompressor compressor(inputImage, centroids, input.cmdOptionExists("-x"));
		compressor.compress();
		Image *outputImage = compressor.getOutputImage();

		kmic::saveRawImage(outputImage, config.getOfn());

		delete inputImage;
		delete outputImage;
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << '\n';
	}

	return 0;
}