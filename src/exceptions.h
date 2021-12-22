#ifndef K_MEANS_IMAGE_COMPRESSOR_EXCEPTION
#define K_MEANS_IMAGE_COMPRESSOR_EXCEPTION
#include <stdexcept>
namespace kmic
{
    class MissingRequiredInputs : public std::exception
    {
    private:
        std::string message;
        void printHelp(const char *_prgName)
        {
            std::string prgName(_prgName);
            message = "Missing required input parameters\n\n";
            message += "USAGE: " + prgName + " [required options] [optional options]\n\nREQUIRED OPTIONS:\n\n";
            message += "-i <Input File Name>\t\tThe image file you want to process.\n";
            message += "-o <Output File Name>\t\tThe name of the output/processed image.\n";
            message += "-k <Number of Centroids>\tThe number of colors to compress the input image down to.\n\n";
            message += "USAGE:   " + prgName + " -i <InputFile> -o <OutputFileName> -k <Number of Centroids>\n";
            message += "EXAMPLE: " + prgName + " -i myImage.png -o outputImage.png -k 2\n\n";
            message += "OPTIONAL OPTIONS:\n\n-x enable X Mode - this mode is slower but may produce better looking results.\n";
        }

    public:
        MissingRequiredInputs(const char *prgName)
        {
            printHelp(prgName);
        }
        const char *what() const noexcept override
        {
            return message.c_str();
        }
    };

    class DecodeImageError : public std::exception
    {
        std::string message = "Decode error: ";

    public:
        DecodeImageError(const char *msg)
        {
            message += std::string(msg);
        }
        const char *what() const noexcept override
        {
            return message.c_str();
        }
    };

    class ConfigError : public std::exception
    {
        std::string message = "Configuration error: ";

    public:
        ConfigError(const char *msg, const std::string &PROGRAM_NAME)
        {
            message += std::string(msg);
            if (PROGRAM_NAME != "")
            {
                message += "\nUSAGE: " + PROGRAM_NAME + " -i <InputFileName> -o <OutputFileName> -k <Number of Centroids>\n";
                message += "\nUse:  '" + PROGRAM_NAME + "' (with no arguments) to see help documentation\n";
            }
        }
        const char *what() const noexcept override
        {
            return message.c_str();
        }
    };
    //InvalidImageType
    class InvalidImageType : public std::exception
    {
        std::string message = "Configuration error: ";

    public:
        InvalidImageType(const char *msg)
        {
            message += std::string(msg);
            message += "/nOnly png and jpeg/jpg images are supported\n";
        }
        const char *what() const noexcept override
        {
            return message.c_str();
        }
    };
}
#endif