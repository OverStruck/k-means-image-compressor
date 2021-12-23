#default compiler flags
CFLAGS = -std=c++17
#source directory
SOURCE_DIR = src/
#type of build: debug or release
BUILD_TYPE = debug
#build directory
BUILD_DIR = build/$(BUILD_TYPE)/
#list of all object files
USER_OBJS = $(wildcard $(BUILD_DIR)*.o)
CSP = $(SOURCE_DIR)dep/ColorSpace/src/
LIBS = -ljpeg -lpng
#default rule
all: check prep release

#-------------------------------
#=======DEBUG SETUP=============
#-------------------------------
#debug compiler flags
debug: CFLAGS += -Wall -Werror -pedantic -Wextra -g
#debug rule
debug: check prep kMeansImageCompressor

#-------------------------------
#=======RELEASE SETUP===========
#-------------------------------
#release compiler flags
release: CFLAGS += -O3 -Wall -DNDEBUG
#set build type to release instead of debug
release: BUILD_TYPE = release
#release rule
release: check prep kMeansImageCompressor

default: kMeansImageCompressor

Conversion:
	g++ $(CFLAGS) -c $(CSP)Conversion.cpp -o $(BUILD_DIR)Conversion.o

Comparison: Conversion
	g++ $(CFLAGS) -c $(CSP)Comparison.cpp -o $(BUILD_DIR)Comparison.o

ColorSpace: Comparison
	g++ $(CFLAGS) -c $(CSP)ColorSpace.cpp -o $(BUILD_DIR)ColorSpace.o

lodepng:
	g++ $(CFLAGS) -c $(SOURCE_DIR)dep/lodepng/lodepng.cpp -o $(BUILD_DIR)lodepng.o

main.o: lodepng ColorSpace
	g++ $(CFLAGS) -c $(SOURCE_DIR)main.cpp -o $(BUILD_DIR)main.o

kMeansImageCompressor: main.o
	g++ $(CFLAGS) $(USER_OBJS) $(LIBS) -o kMeansImageCompressor

prep:
	mkdir -p build/debug build/release

check:
#check if jpeg and png libraries exits in system
ifeq (none,$(strip $(shell apt-cache policy libjpeg-dev | grep -Po '(?<=\().*(?=\))')))
	$(error "No jpeg library (ljpeg) found in your system, consider doing apt install libjpeg-dev")
endif

ifeq (none,$(strip $(shell apt-cache policy libpng-dev | grep -Po '(?<=\().*(?=\))')))
	$(error "No png library (lpng) found in your system, consider doing apt install libpng-dev")
endif

clean:
	rm -r build
	rm -f kMeansImageCompressor
