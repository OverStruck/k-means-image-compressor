#ifndef K_MEANS_CENTROIDS
#define K_MEANS_CENTROIDS
#include <cstdint>
#include <sstream>
class Centroids
{
public:
    uint8_t size = 0;
    uint8_t *red, *green, *blue;

    Centroids(){};
    Centroids(const uint8_t numberOfCentroids)
    {
        init(numberOfCentroids);
    }

    void init(const uint8_t numberOfCentroids)
    {
        size = numberOfCentroids;
        red = new uint8_t[size];
        green = new uint8_t[size];
        blue = new uint8_t[size];
    }

    void copy(const Centroids &other)
    {
        for (uint8_t i = 0; i < size; i++)
        {
            red[i] = other.red[i];
            green[i] = other.green[i];
            blue[i] = other.blue[i];
        }
    }

    void clear()
    {
        for (uint8_t i = 0; i < size; i++)
        {
            red[i] = 0;
            green[i] = 0;
            blue[i] = 0;
        }
    }

    void printColor(int index)
    {
        std::ostringstream colorBar;

        const int r = static_cast<int>(red[index]);
        const int g = static_cast<int>(green[index]);
        const int b = static_cast<int>(blue[index]);

        colorBar << "\x1b[38;2;" << r << ";" << g << ";" << b << "m"
                 << "█████████████████"
                 << "\x1b[0m";

        printf("Centroid[%d]\t= [%3d, %3d, %3d]\t=> %s\n", index + 1, r, g, b, colorBar.str().c_str());
    }

    void print()
    {
        for (uint8_t i = 0; i < size; i++)
        {
            printColor(i);
        }
    }

    ~Centroids()
    {
        delete[] red;
        delete[] green;
        delete[] blue;
    }
};
#endif