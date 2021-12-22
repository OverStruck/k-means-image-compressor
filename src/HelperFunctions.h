#ifndef K_MEANS_IMAGE_COMPRESSOR_HELPER_FUNCS
#define K_MEANS_IMAGE_COMPRESSOR_HELPER_FUNCS

#include "exceptions.h"
#include "Image.h"
#include "Centroids.h"
#include "dep/lodepng/lodepng.h"

#include "exceptions.h"
#include <bits/stdc++.h>

#define cimg_display 0
#include "dep/CImg/CImg.h"

namespace kmic
{
    Image *loadJPGImage(const char *inputFileName)
    {
        printf("Loading image...\n");
        unsigned int width, height;
        cimg_library::CImg<unsigned char> image(inputFileName);
        // image.blur(1);
        width = image.width();
        height = image.height();
        Image *inputImage = new Image(width, height);
        size_t where = 0;
        for (unsigned int y = 0; y < height; y++)
        {
            for (unsigned int x = 0; x < width; x++)
            {
                inputImage->red[where] = static_cast<uint8_t>(image(x, y, 0, 0));
                inputImage->green[where] = static_cast<uint8_t>(image(x, y, 0, 1));
                inputImage->blue[where] = static_cast<uint8_t>(image(x, y, 0, 2));
                where++;
            }
        }
        printf("Image loaded: (%zu x %zu)\n\n", inputImage->width, inputImage->height);
        return inputImage;
    }

    Image *loadPNGImage(const char *inputFileName)
    {
        printf("Loading image...\n");
        unsigned int width, height;
        std::vector<unsigned char> inputImageRawPixels;
        unsigned error = lodepng::decode(inputImageRawPixels, width, height, inputFileName, LCT_RGB);
        if (error)
        {
            throw DecodeImageError(lodepng_error_text(error));
        }

        Image *inputImage = new Image(width, height);

        size_t where = 0;
        for (size_t i = 0; i < inputImage->size * 3; i += 3)
        {
            inputImage->red[where] = inputImageRawPixels.at(i);
            inputImage->green[where] = inputImageRawPixels.at(i + 1);
            inputImage->blue[where] = inputImageRawPixels.at(i + 2);
            ++where;
        }
        printf("Image loaded: (%zu x %zu)\n\n", inputImage->width, inputImage->height);
        return inputImage;
    }

    Image *loadRawImage(const std::string &inputFileName)
    {
        std::string fileExtension = inputFileName.substr(inputFileName.find_last_of(".") + 1);
        if (fileExtension == "png")
        {
            return loadPNGImage(inputFileName.c_str());
        }
        else if (fileExtension == "jpg" || fileExtension == "jpeg")
        {
            printf("jpg image detected\n");
            return loadJPGImage(inputFileName.c_str());
        }
        else
        {
            throw InvalidImageType("Invalid input file");
        }
    }

    void saveRawImage(Image *&outputImagePixels, const char *outputFileName)
    {
        printf("\nWriting compressed image...\n");
        const unsigned int width = outputImagePixels->width;
        const unsigned int height = outputImagePixels->height;
        cimg_library::CImg<unsigned char> outputImage(width, height, 1, 3);

        size_t where = 0;
        for (unsigned int y = 0; y < height; y++)
        {
            for (unsigned int x = 0; x < width; x++)
            {
                outputImage(x, y, 0, 0) = outputImagePixels->red[where];
                outputImage(x, y, 0, 1) = outputImagePixels->green[where];
                outputImage(x, y, 0, 2) = outputImagePixels->blue[where];
                where++;
            }
        }
        outputImage.save(outputFileName);
        printf("Image '%s' saved to disk\n", outputFileName);
    }

    void savePNGImage(Image *&outputImagePixels, const char *outputFileName)
    {
        printf("\nWriting compressed image...\n");

        // Prepare data for output
        std::vector<unsigned char> outputImage;
        for (unsigned int i = 0; i < outputImagePixels->size; ++i)
        {
            outputImage.push_back(outputImagePixels->red[i]);
            outputImage.push_back(outputImagePixels->green[i]);
            outputImage.push_back(outputImagePixels->blue[i]);
        }

        // Output the data
        unsigned error = lodepng::encode(outputFileName, outputImage, outputImagePixels->width, outputImagePixels->height, LCT_RGB);

        // //if there's an error, display it
        if (error)
        {
            std::cout << "Encoder error: " << error << ": " << lodepng_error_text(error) << std::endl;
            exit(-1);
        }
        printf("Image '%s' saved to disk\n", outputFileName);
    }

    void initUserCentroids(Centroids &centroids)
    {
        uint8_t colors[4][3] = {
            {38, 26, 56},
            {254, 241, 201},
            {126, 153, 137},
            {253, 242, 211}};

        for (uint8_t i = 0; i < centroids.size; i++)
        {
            centroids.red[i] = colors[i][0];
            centroids.green[i] = colors[i][1];
            centroids.blue[i] = colors[i][2];
        }
        printf("Initial Centroids:\n");
        centroids.print();
    }

    void initCentroids(Centroids &centroids, Image *&inputImage)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> xDis(0, inputImage->width);
        std::uniform_int_distribution<> yDis(0, inputImage->height);
        unsigned int x, y, index;
        printf("Initializing %d centroids at random\n\n", static_cast<int>(centroids.size));
        for (uint8_t i = 0; i < centroids.size; i++)
        {
            x = xDis(gen);
            y = yDis(gen);
            index = x * y;
            centroids.red[i] = inputImage->red[index];
            centroids.green[i] = inputImage->green[index];
            centroids.blue[i] = inputImage->blue[index];
        }
        printf("Initial Centroids:\n");
        centroids.print();
    }

    void printWelcome()
    {
        printf("\n=============K MEANS IMAGE COMPRESSOR=============\n\n");
    }
}

#endif