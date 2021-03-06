DEPS=activation.h  conv.h  cppnn.h  error.h  image_utils.h  layer.h  mat.h  misc.h  tensor.h  test_conv.cu  test.cu  CImg.hh

# dummy
all: format test test_conv cuda

cuda: format test_cuda test_conv_cuda

format: $(DEPS)
	@clang-format-3.8 -i *.h *.cu

test: $(DEPS)
	@nvcc --std=c++11 $@.cu -o $@ -g -Xcompiler -fopenmp -lgomp

test_conv: $(DEPS)
	@nvcc --std=c++11 $@.cu -o $@ -lpthread -lX11 -w -g -Xcompiler -fopenmp -lgomp

test_cuda: $(DEPS)
	@nvcc --std=c++11 test.cu -o $@ -w -D CUDA -g -Xcompiler -fopenmp -lgomp

test_conv_cuda: $(DEPS)
	@nvcc --std=c++11 test_conv.cu -o $@ -lpthread -lX11 -w -D CUDA -g -Xcompiler -fopenmp -lgomp

clean:
	rm -f test
	rm -f test_conv
	rm -f test_cuda
	rm -f test_conv_cuda
