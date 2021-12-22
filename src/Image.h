#ifndef K_MEANS_IMAGE_COMPRESSOR_IMAGE
#define K_MEANS_IMAGE_COMPRESSOR_IMAGE
#include <cstdint>
#include <cstddef>
class Image
{
public:
   uint8_t *red, *green, *blue;
    size_t width, height, size;

    Image(const size_t w, const size_t h)
    {
        width = w;
        height = h;
        size = w * h;
        red = new uint8_t[size];
        green = new uint8_t[size];
        blue = new uint8_t[size];
    }

    ~Image()
    {
        delete[] red;
        delete[] green;
        delete[] blue;
    }
};

#endif