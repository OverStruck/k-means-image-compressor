#ifndef K_MEANS_IMAGE_COMPRESSOR
#define K_MEANS_IMAGE_COMPRESSOR
#include "Image.h"
#include "Centroids.h"
#include "RgbSums.h"
#include "dep/ColorSpace/src/Comparison.h"

#include <cmath>
#include <limits>
#include <iostream>

#include <chrono>

namespace kmic
{
    class ImageCompressor
    {
    private:
        bool Cie2000Comparison = false;
        Image *inputImage;
        uint8_t numberOfCentroids;

        Centroids centroids;
        Centroids prevCentroids;
        RgbSums rgbSums;

        size_t *labelArray;
        unsigned int *pixelClusterCounter;

        void process()
        {
            for (uint8_t i = 0; i < numberOfCentroids; i++)
            {
                pixelClusterCounter[i] = 0;
            }

            if (Cie2000Comparison)
                assignLabels2();
            else
                assignLabels();

            sumClusters();
            generateNewCentroids();
        }
        void assignLabels()
        {
            for (size_t j = 0; j < inputImage->size; j++)
            {
                size_t label = 0;
                float euclideanDistance, currentDistance = std::numeric_limits<float>::max();
                for (uint8_t i = 0; i < numberOfCentroids; i++)
                {
                    euclideanDistance = getEuclideanDistance(j, i);

                    if (euclideanDistance < currentDistance)
                    {
                        currentDistance = euclideanDistance;
                        label = i;
                    }
                }
                labelArray[j] = label;
            }
        }
        void assignLabels2()
        {
            for (size_t j = 0; j < inputImage->size; j++)
            {
                size_t label = 0;
                double euclideanDistance, currentDistance = std::numeric_limits<double>::max();
                ColorSpace::Rgb c1(inputImage->red[j], inputImage->green[j], inputImage->blue[j]);
                for (uint8_t i = 0; i < numberOfCentroids; i++)
                {
                    ColorSpace::Rgb c2(centroids.red[i], centroids.green[i], centroids.blue[i]);
                    euclideanDistance = ColorSpace::Cie2000Comparison::Compare(&c1, &c2);
                    if (euclideanDistance < currentDistance)
                    {
                        currentDistance = euclideanDistance;
                        label = i;
                    }
                }
                labelArray[j] = label;
            }
        }
        float getEuclideanDistance(const size_t j, const uint8_t i)
        {
            const int rd = inputImage->red[j] - centroids.red[i];
            const int gd = inputImage->green[j] - centroids.green[i];
            const int bd = inputImage->blue[j] - centroids.blue[i];
            return sqrtf((powf(rd, 2.0) + powf(gd, 2.0) + powf(bd, 2.0)));
        }
        void sumClusters()
        {
            rgbSums.clear();
            for (size_t j = 0; j < inputImage->size; j++)
            {
                const unsigned int clusterLabel = labelArray[j];
                const unsigned int currentRed = static_cast<unsigned int>(inputImage->red[j]);
                const unsigned int currentGreen = static_cast<unsigned int>(inputImage->green[j]);
                const unsigned int currentBlue = static_cast<unsigned int>(inputImage->blue[j]);
                rgbSums.red[clusterLabel] += currentRed;
                rgbSums.green[clusterLabel] += currentGreen;
                rgbSums.blue[clusterLabel] += currentBlue;
                pixelClusterCounter[clusterLabel] += 1;
            }
        }
        void generateNewCentroids()
        {
            for (unsigned int i = 0; i < numberOfCentroids; i++)
            {
                const unsigned int currentPixelCounter = pixelClusterCounter[i];
                if (currentPixelCounter > 0)
                {
                    const unsigned int sumRed = rgbSums.red[i];
                    const unsigned int sumGreen = rgbSums.green[i];
                    const unsigned int sumBlue = rgbSums.blue[i];
                    const uint8_t redAverage = static_cast<uint8_t>((sumRed / currentPixelCounter));
                    const uint8_t greenAverage = static_cast<uint8_t>((sumGreen / currentPixelCounter));
                    const uint8_t blueAverage = static_cast<uint8_t>((sumBlue / currentPixelCounter));
                    centroids.red[i] = redAverage;
                    centroids.green[i] = greenAverage;
                    centroids.blue[i] = blueAverage;
                }
            }
        }
        float getEuclideanDistance(int i)
        {
            const int rd = prevCentroids.red[i] - centroids.red[i];
            const int gd = prevCentroids.green[i] - centroids.green[i];
            const int bd = prevCentroids.blue[i] - centroids.blue[i];
            return sqrtf((powf(rd, 2.0) + powf(gd, 2.0) + powf(bd, 2.0)));
        }
        bool areCentroidsSame()
        {
            int same = false;
            for (uint8_t i = 0; i < numberOfCentroids; i++)
            {
                if (getEuclideanDistance(i) > 30)
                {
                    same = true;
                    break;
                }
            }
            return same;
        }

    public:
        ImageCompressor(Image *_image, const Centroids &_centroids, const bool _Cie2000Comparison)
        {
            Cie2000Comparison = _Cie2000Comparison;
            inputImage = _image;
            numberOfCentroids = _centroids.size;
            centroids.init(_centroids.size);
            prevCentroids.init(_centroids.size);
            rgbSums.init(_centroids.size);
            centroids.copy(_centroids);
            pixelClusterCounter = new unsigned int[_centroids.size];
            labelArray = new size_t[_image->size];
        }
        void compress()
        {
            int iterCount = 0;
            auto tStart = std::chrono::high_resolution_clock::now();
            do
            {
                prevCentroids.copy(centroids);
                process();
                iterCount++;
            } while (areCentroidsSame());
            printf("\nDone compressing image. Total iterations: %d\n", iterCount);
            auto tEnd = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
            std::cout << "It took " << duration << " milliseconds to do this work\n\n";

            printf("Final Centroids:\n");
            centroids.print();
        }

        Image *getOutputImage()
        {
            Image *outputImage = new Image(inputImage->width, inputImage->height);
            for (size_t i = 0; i < inputImage->size; ++i)
            {
                outputImage->red[i] = centroids.red[labelArray[i]];
                outputImage->green[i] = centroids.green[labelArray[i]];
                outputImage->blue[i] = centroids.blue[labelArray[i]];
            }
            return outputImage;
        }

        ~ImageCompressor()
        {
            delete[] labelArray;
            delete[] pixelClusterCounter;
        }
    };
}

#endif