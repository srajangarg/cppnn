# dummy
all: format test test_conv

cuda: format test_cuda test_conv_cuda

format:
	@clang-format-3.8 -i *.h *.cu

test:
	@nvcc test.cu -std=c++11 -O3 -otest -Wextra -Wpedantic

test_conv:
	@nvcc --std=c++11 test_conv.cu -lpthread -lX11 -otest_conv -w

test_cuda:
	@nvcc test.cu -std=c++11 -O3 -otest -Wextra -Wpedantic -D CUDA

test_conv_cuda:
	@nvcc --std=c++11 test_conv.cu -lpthread -lX11 -otest_conv -w -D CUDA

clean:
	rm test
	rm test_conv
