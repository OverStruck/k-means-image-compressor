#ifndef K_MEANS_RGB_SUMS
#define K_MEANS_RGB_SUMS
#include <cstdint>
class RgbSums
{
public:
    uint8_t size = 0;
    unsigned int *red = nullptr, *green = nullptr, *blue = nullptr;
    RgbSums(){};
    RgbSums(const uint8_t numberOfCentroids)
    {
        init(numberOfCentroids);
    }
    void init(const uint8_t numberOfCentroids)
    {
        size = numberOfCentroids;
        red = new unsigned int[size];
        green = new unsigned int[size];
        blue = new unsigned int[size];
    }
    void clear()
    {
        for (uint8_t i = 0; i < size; i++)
        {
            red[i] = 0u;
            green[i] = 0u;
            blue[i] = 0u;
        }
    }
    ~RgbSums()
    {
        delete[] red;
        delete[] green;
        delete[] blue;
    }
};
#endif