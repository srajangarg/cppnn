#pragma once

#define NDIM_MAX 4

#include <stdarg.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

__global__ tensor_add_(float* lhs, float* rhs, int size, float scale) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < size)
		lhs[index] += scale*rhs[index];
}
__global__ tensor_mul_(float* tensor, int size, float c) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < size)
		tensor[index] *= c;
}

class Tensor {
public:
	float* data;
	unsigned long shape[NDIM_MAX];
	bool is_cuda;
	int ndims;
	bool ndims() {return ndims;}

private:
	void init(int n, ...) {
		assert(ndims == n-1); // Already done in constructor
		assert(ndims <= NDIM_MAX);
		va_list vl;
		va_start(vl,n);
		data = va_arg(vl,float*);
		for (int i = 0; i < n; ++i)
			shape[i] = va_arg(vl,int);
		for (int i = n; i < NDIM_MAX; ++i)
			shape[i] = 1;
		va_end(vl);
		if(data == NULL)
			data = new float[numel()];
		assert(data != NULL);
	};

public:
	Tensor():is_cuda(false),data(NULL),ndims(0){};
	Tensor(int H):Tensor(),ndims(1){
		init(2, (float*)NULL, H);
	}
	Tensor(int H, int W):Tensor(),ndims(2){
		init(3, (float*)NULL, H, W);
	}
	Tensor(int H, int W, int C):Tensor(),ndims(3){
		init(4, (float*)NULL, H, W, C);
	}
	Tensor(int H, int W, int C, int D):Tensor(),ndims(4){
		init(5, (float*)NULL, H, W, C, D);
	}
	// Does not copy data from float
	Tensor(float*dat, int H):Tensor(),ndims(1){
		init(2, dat, H);
	}
	Tensor(float*dat, int H, int W):Tensor(),ndims(2){
		init(3, dat, H, W);
	}
	Tensor(float*dat, int H, int W, int C):Tensor(),ndims(3){
		init(4, dat, H, W, C);
	}
	Tensor(float*dat, int H, int W, int C, int D):Tensor(),ndims(4){
		init(5, dat, H, W, C, D);
	}
	~Tensor() {
		if(data != NULL) {
			delete []data;
			data = NULL;
		}
	}

	Tensor& operator=(Tensor&other) {
		if(data != NULL) {
			delete [] data;
			data = NULL;
		}
		data = other.data;
		for (int i = 0; i < NDIM_MAX; ++i) {
			shape[i] = other.shape[i];
		}
		is_cuda = other.is_cuda;
		ndims = other.ndims;
	}

	Tensor(Tensor&other) {
		if(data != NULL) {
			delete [] data;
			data = NULL;
		}
		data = other.data;
		for (int i = 0; i < NDIM_MAX; ++i) {
			shape[i] = other.shape[i];
		}
		is_cuda = other.is_cuda;
		ndims = other.ndims;
	}

	unsigned long numel() {
		unsigned long res = 1;
		for (int i = 0; i < ndims; ++i)
			res *= shape[i];
		return res;
	}

	void cuda() {
		if(!is_cuda) {
			float* cuda_data = NULL;
			cudaMalloc((void **)&cuda_data, numel() * sizeof(float));
			cudaMemcpy(cuda_data, data, numel() * sizeof(float), cudaMemcpyHostToDevice);
			delete []data;
			data = cuda_data;
		}
	}

// ACCESS
	float item() {
		assert(numel() == 1);
		if(is_cuda) {
			float res;
			cudaMemcpy(&res, data, sizeof(float), cudaMemcpyDeviceToHost);
			return res;
		}
		else
			return data[0];
	}
	float & at(int i, int j, int k, int l) {
		assert(is_cuda == false);
		assert(ndims() == 4);
		int index = shape[3]*( shape[2]*( shape[1]*i+j )+k )+l;
		return data[index];
	}
	float & at(int i, int j, int k) {
		assert(is_cuda == false);
		assert(ndims() == 3);
		int index = shape[2]*( shape[1]*i+j )+k;
		return data[index];
	}
	float & at(int i, int j) {
		assert(is_cuda == false);
		assert(ndims() == 2);
		int index = shape[1]*i+j;
		return data[index];
	}
	float & at(int i) {
		assert(is_cuda == false);
		assert(ndims() == 1);
		int index = i;
		return data[index];
	}

	void add_(Tensor&other, float scale=1) {
		assert(is_cuda == other.is_cuda);
		assert(numel() == other.numel());
		if(is_cuda) {
			int num_blocks = (numel()+1)/THREADS_PER_BLOCK;
			tensor_add_<<<num_blocks, THREADS_PER_BLOCK>>>(data, other.data, numel(), scale);
		}
		else {
			for (int i = 0; i < numel(); ++i)
				data[i] += scale*other.data[i];
		}
	}
	void copy_(Tensor&other) {
		assert(is_cuda == other.is_cuda);
		assert(numel() == other.numel());
		if(is_cuda)
			cudaMemcpy(data, other.data, numel() * sizeof(float), cudaMemcpyDeviceToDevice);
		else
			memcpy(data, other.data, numel() * sizeof(float));
	}
	void fill_(float c) {
		if(is_cuda)
			cudaMemset(data, c, numel()*sizeof(float))
		else
			std::fill(data, data+numel(), c);
	}
	void mul_(float c) {
		if(is_cuda) {
			int num_blocks = (numel()+1)/THREADS_PER_BLOCK;
			tensor_mul_<<<num_blocks, THREADS_PER_BLOCK>>>(data, numel(), c);
		}
		else {
			for (int i = 0; i < numel(); ++i)
				data[i] *= c;
		}
	}
	int arg_minmax(bool max) {
		float *cpu_data = NULL;
		if(is_cuda) {
			// TODO: Better Reduction Method
			cpu_data = new float[numel()];
			cudaMemcpy(cpu_data, data, numel() * sizeof(float), cudaMemcpyDeviceToHost);
		}
		else {
			cpu_data = data;
		}
		int max = 0;
		for (int i = 0; i < numel(); ++i)
			if((cpu_data[i] < cpu_data[max]) ^ max)
				max = i;

		if(is_cuda)
			delete []cpu_data;

		return max;
	}
	int argmax() {return arg_minmax(true);}
	int argmin() {return arg_minmax(false);}
}
