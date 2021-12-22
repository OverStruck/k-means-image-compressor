#ifndef K_MEANS_IMAGE_COMPRESSOR_CONFIG
#define K_MEANS_IMAGE_COMPRESSOR_CONFIG

#include "exceptions.h"
#include "InputParser.h"

#include <string>
#include <sys/stat.h>
#include <algorithm>
#include <cstdarg>

namespace kmic
{
    void LOG_IF(const bool condition, const char *format, ...)
    {
        if (condition)
        {
            va_list argptr;
            va_start(argptr, format);
            vfprintf(stderr, format, argptr);
            va_end(argptr);
        }
    }

    class Config
    {
    private:
        std::string ifn;
        std::string ofn;
        uint8_t k;

        bool fileExist(const std::string &fileName)
        {
            struct stat buffer;
            return (stat(fileName.c_str(), &buffer) == 0);
        }

        uint8_t parseK(const std::string &number)
        {
            int thisK;
            try
            {
                thisK = std::stoi(number);
                //warning message if k is negative
                LOG_IF(thisK < 0, "Negative numbers are NOT allowed for 'k' parameter\n");

                //warning message if k is less than 2
                LOG_IF(thisK < 2, "'k' parameter value is too low!\n");
                LOG_IF(thisK < 2, "'k' is %d => Using k=2 - the mininum allowed value\n\n", thisK);

                //warning message if k is greater than 255
                LOG_IF(thisK > 255, "'k' parameter value is too high!\n");
                LOG_IF(thisK > 255, "'k' is %d => Using k=255 - the max allowed value\n\n", thisK);
                //return number between 2 and 255
                return static_cast<uint8_t>(std::clamp(thisK, 2, 255));
            }
            catch (...)
            {
                printf("Invalid 'k' parameter used. Setting k=2\n");
                return static_cast<uint8_t>(2);
            }
        }

    public:
        bool errors = false;
        Config() {}
        Config(const InputParser &input)
        {
            std::string prgName = input.getCmdOption("prgName");

            ifn = input.getCmdOption("-i");
            //check input file
            if (ifn.empty())
            {
                throw ConfigError("Missing REQUIRED input file name", prgName);
            }

            if (!fileExist(ifn))
            {
                prgName = ""; //lazy way
                throw ConfigError("Input file does NOT exists!", prgName);
            }

            //initialize required config args
            ofn = input.getCmdOption("-o");
            //check output file names
            if (ofn.empty())
            {
                throw ConfigError("Missing REQUIRED output file name", prgName);
            }

            std::string fe = ofn.substr(ofn.find_last_of(".") + 1);
            if (fe != "png" && fe != "jpg" && fe != "jpeg")
            {
                prgName = ""; //lazy way
                throw ConfigError("Invalid output file name!\nOnly png/jpg/jpeg extensions are supported", prgName);
            }

            k = parseK(input.getCmdOption("-k"));
        }
        //getters
        const char *getIfn()
        {
            return ifn.c_str();
        }

        const char *getOfn()
        {
            return ofn.c_str();
        }

        const std::string &getifn()
        {
            return ifn;
        }

        uint8_t getK()
        {
            return k;
        }
    };
}
#endif