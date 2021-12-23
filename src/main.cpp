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
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
    catch (...)
    {
        std::exception_ptr p = std::current_exception();
        std::clog << (p ? p.__cxa_exception_type()->name() : "null") << std::endl;
        std::cerr << "Unknown error happened :(" << std::endl;
    }
    return 0;
}