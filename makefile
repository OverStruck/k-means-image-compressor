
CFLAGS = -std=c++11
CUDA_FLAGS = -lcuda -lcudart
LODEPNG = dep/lodepng/
CSP = dep/ColorSpace/src/

default: kmeansCompressor

lodepng.o:
	nvcc -dc $(LODEPNG)lodepng.cpp -o lodepng.o
	
Conversion.o: 
	g++ -c $(CSP)Conversion.cpp -o Conversion.o

Comparison.o: Conversion.o
	g++ -c $(CSP)Comparison.cpp -o Comparison.o
	
ColorSpace.o: Comparison.o
	g++ -c $(CSP)ColorSpace.cpp -o ColorSpace.o
	
main.o:
	g++ $(CFLAGS) -c main.cpp -o main.o $(CUDA_FLAGS)
	
kmeansCompressor2: lodepng.o ColorSpace.o kmeansCompressor.o main.o
	nvcc  $(CFLAGS) main.o kmeansCompressor.o lodepng.o ColorSpace.o Comparison.o Conversion.o -o kmeansCompressor2 $(CUDA_FLAGS)
	
kmeansCompressor.o:
	nvcc $(CFLAGS) kmeansCompressor.cu -dc -o kmeansCompressor.o  $(CUDA_FLAGS)
	
kmeansCompressor: lodepng.o kmeansCompressor.o
	nvcc $(CFLAGS) kmeansCompressor.o lodepng.o -o kmeansCompressor

clean:
	rm -f *.o kmeansCompressor