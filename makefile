# dummy
all:

	g++ test.cu -std=c++11 -O3 -otest

cuda:
	@clang-format-3.8 -i *.h *.cpp
	@nvcc test.cpp -o test
clean:

	rm test
