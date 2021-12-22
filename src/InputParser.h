#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H
#include "exceptions.h"
#include <string>
#include <vector>
#include <algorithm>
namespace kmic
{
    class InputParser
    {
    private:
        std::vector<std::string> tokens;

    public:
        InputParser(int &argc, char **argv)
        {
            if (argc == 1)
            {
                throw MissingRequiredInputs(argv[0]);
            }
            //store program name
            this->tokens.push_back("prgName");
            this->tokens.push_back(std::string(argv[0]));
            //store rest of arguments
            for (int i = 1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        const std::string &getCmdOption(const std::string &option) const
        {
            std::vector<std::string>::const_iterator itr;
            itr = std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end())
            {
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        bool cmdOptionExists(const std::string &option) const
        {
            return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
        }
    };
}
#endif