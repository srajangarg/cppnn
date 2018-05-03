# dummy
all: format test test_conv

format:
	@clang-format-3.8 -i *.h *.cu

test:
	@nvcc test.cpp -std=c++11 -O3 -otest -Wextra -Wpedantic

test_conv:
	@nvcc --std=c++11 test_conv.cu -lpthread -lX11 -otest_conv -w

clean:
	rm test
	rm test_conv
