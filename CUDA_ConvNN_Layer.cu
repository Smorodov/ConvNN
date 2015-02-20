#include "CUDA_ConvNN_Layer.h"
#include <cstdio>
#include <cassert>
#include "CUDA_Common.h"

using namespace std;

// Uncomment this if you're calling CheckValues()
//#define NO_CHECK

namespace CNN
{

	__global__ void DeviceSetValue(REAL *weight, int idx, REAL v)
	{
		weight[idx] = v;
	}

	__global__ void DeviceGetValue(REAL *weight, int idx, REAL *v)
	{
		*v = weight[idx];
	}

	__global__ void SumGrads(const REAL *weight_grad, int weight_size, int num_images, REAL *sum_weight)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if(idx >= weight_size)
		{
			return;
		}
		REAL sum=0;
		for(int i=0; i < num_images; i++)
		{
			sum += weight_grad[i*weight_size + idx];
		}
		sum_weight[idx] = sum;
	}

	Layer::Layer()
	{
		num_images = 0;
		in_image_width = 0;
		out_image_width = 0;
		in_channels = 0;
		out_channels = 0;
		conv_size = 0;
		in_size = 0;
		out_size = 0;
		weight_size = 0;
		bias_size = 0;
		out_total_size = 0;
		out_size_per_sample = 0;
		weight_rows = 0;
		weight_cols = 0;
		grad_size = 0;
		mask_size = 0;
		total_mem_used = 0;
		d_weights = NULL;
		d_biases = NULL;
		d_out = NULL;
		d_out_deriv = NULL;
		d_grad = NULL;
		d_mask = NULL;
		d_sum_weight_grad = NULL;
		d_sum_bias_grad = NULL;
		d_delta_weight = NULL;
		d_delta_bias = NULL;
		d_momentum_delta_weight = NULL;
		d_momentum_delta_bias = NULL;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_tmp, sizeof(REAL)));
	}

	void Layer::CleanUp()
	{
		if(d_weights)
		{
			cudaFree(d_weights);
		}
		if(d_biases)
		{
			cudaFree(d_biases);
		}
		if(d_out)
		{
			cudaFree(d_out);
		}
		if(d_out_deriv)
		{
			cudaFree(d_out_deriv);
		}
		if(d_grad)
		{
			cudaFree(d_grad);
		}
		if(d_mask)
		{
			cudaFree(d_mask);
		}
		if(d_sum_weight_grad)
		{
			cudaFree(d_sum_weight_grad);
		}
		if(d_sum_bias_grad)
		{
			cudaFree(d_sum_bias_grad);
		}
		if(d_delta_weight)
		{
			cudaFree(d_delta_weight);
		}
		if(d_delta_bias)
		{
			cudaFree(d_delta_bias);
		}
		if(d_momentum_delta_weight)
		{
			cudaFree(d_momentum_delta_weight);
		}
		if(d_momentum_delta_bias)
		{
			cudaFree(d_momentum_delta_bias);
		}
		if(d_tmp)
		{
			cudaFree(d_tmp);
		}
	}

	void Layer::InitConv(LayerType t)
	{
		assert(in_image_width > 0);
		assert(in_channels > 0);
		assert(out_channels > 0);
		assert(conv_size > 0);
		assert(num_images > 0);
		assert(conv_size <= in_image_width);
		type = t;
		out_image_width = in_image_width - conv_size + 1;
		assert(out_image_width > 0);
		weight_size = conv_size*conv_size*in_channels*out_channels;
		bias_size = in_channels*out_channels;
		out_size_per_sample = out_image_width*out_image_width*out_channels;
		out_total_size = num_images*out_size_per_sample;
		grad_size = out_total_size;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_deriv, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_grad, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_weights, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_delta_weight, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_momentum_delta_weight, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_sum_weight_grad, weight_size*num_images*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_biases, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_delta_bias, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_momentum_delta_bias, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_sum_bias_grad, bias_size*num_images*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_biases, 0, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_momentum_delta_weight, 0, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_momentum_delta_bias, 0, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_out_deriv, 0, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_grad, 0, out_total_size*sizeof(REAL)));
		total_mem_used = out_total_size*3 + weight_size*3 + weight_size*num_images + bias_size*3 + bias_size*num_images;
		total_mem_used *= sizeof(REAL);
		ZeroOutput();
		ZeroSumGrad();
	}

	void Layer::InitPool(LayerType t)
	{
		assert(num_images > 0);
		assert(in_image_width > 0);
		assert(in_channels > 0);
		assert(in_image_width % 2 == 0);
		type = t;
		out_channels = in_channels;
		out_image_width = in_image_width / 2;
		assert(out_image_width > 0);
		out_size_per_sample = out_image_width*out_image_width*out_channels;
		out_total_size = num_images*out_size_per_sample;
		grad_size = out_total_size;
		mask_size = num_images*in_image_width*in_image_width*in_channels;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_mask, mask_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_grad, grad_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_mask, 0, mask_size*sizeof(REAL)));
		total_mem_used = out_total_size + mask_size + grad_size;
		total_mem_used *= sizeof(REAL);
	}

	void Layer::InitNN(LayerType t)
	{
		assert(num_images > 0);
		assert(in_size > 0);
		assert(out_size > 0);
		type = t;
		weight_size = in_size*out_size;
		bias_size = out_size;
		out_total_size = num_images*out_size;
		grad_size = out_total_size;
		weight_rows = out_size;
		weight_cols = in_size;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_deriv, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_grad, out_total_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_weights, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_delta_weight, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_momentum_delta_weight, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_sum_weight_grad, weight_size*num_images*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_biases, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_delta_bias, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_momentum_delta_bias, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_sum_bias_grad, bias_size*num_images*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_biases, 0, bias_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_momentum_delta_weight, 0, weight_size*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_momentum_delta_bias, 0, bias_size*sizeof(REAL)));
		total_mem_used = out_total_size*3 + weight_size*3 + weight_size*num_images + bias_size*3 + bias_size*num_images;
		total_mem_used *= sizeof(REAL);
		ZeroSumGrad();
	}

	void Layer::InitSoftMax()
	{
		assert(num_images > 0);
		assert(in_size > 0);
		type = NN_SOFTMAX;
		out_size = in_size;
		out_total_size = num_images*out_size;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, out_total_size*sizeof(REAL)));
		total_mem_used = out_total_size;
		total_mem_used *= sizeof(REAL);
	}

	void Layer::InitRandWeights(REAL r)
	{
		assert(d_weights);
		assert(weight_size > 0);
		vector <REAL> w(weight_size);
		for(size_t i=0; i < w.size(); i++)
		{
			w[i] = -r + 2*r*(rand()/(1.0+RAND_MAX));
		}
		CUDA_SAFE_CALL(cudaMemcpy(d_weights, &w[0], weight_size*sizeof(REAL), cudaMemcpyHostToDevice));
	}

	void Layer::SetWeights(const REAL *weights)
	{
		assert(d_weights);
		assert(weight_size > 0);
		CUDA_SAFE_CALL(cudaMemcpy(d_weights, weights, weight_size*sizeof(REAL), cudaMemcpyHostToDevice));
	}

	void Layer::SetMomentumWeights(const REAL *weights)
	{
		assert(d_weights);
		assert(weight_size > 0);
		CUDA_SAFE_CALL(cudaMemcpy(d_momentum_delta_weight, weights, weight_size*sizeof(REAL), cudaMemcpyHostToDevice));
	}

	void Layer::SetBiases(const REAL *biases)
	{
		assert(d_biases);
		assert(bias_size > 0);
		CUDA_SAFE_CALL(cudaMemcpy(d_biases, biases, bias_size*sizeof(REAL), cudaMemcpyHostToDevice));
	}

	void Layer::SetMomentumBiases(const REAL *biases)
	{
		assert(d_biases);
		assert(bias_size > 0);
		CUDA_SAFE_CALL(cudaMemcpy(d_momentum_delta_bias, biases, bias_size*sizeof(REAL), cudaMemcpyHostToDevice));
	}


	void Layer::GetWeights(std::vector<REAL> &weights)
	{
		weights.resize(weight_size);
		CUDA_SAFE_CALL(cudaMemcpy(&weights[0], d_weights, weight_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetMomentumWeights(std::vector<REAL> &weights)
	{
		weights.resize(weight_size);
		CUDA_SAFE_CALL(cudaMemcpy(&weights[0], d_momentum_delta_weight, weight_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetBiases(std::vector <REAL> &biases)
	{
		biases.resize(bias_size);
		CUDA_SAFE_CALL(cudaMemcpy(&biases[0], d_biases, bias_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetMomentumBiases(std::vector <REAL> &biases)
	{
		biases.resize(bias_size);
		CUDA_SAFE_CALL(cudaMemcpy(&biases[0], d_momentum_delta_bias, bias_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetOutput(std::vector <REAL> &output)
	{
#ifndef NO_CHECK
		assert(out_total_size > 0);
		assert(d_out);
#endif
		output.resize(out_total_size);
		CUDA_SAFE_CALL(cudaMemcpy(&output[0], d_out, out_total_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetSumWeightGrad(std::vector <REAL> &sum_weight_grad)
	{
#ifndef NO_CHECK
		assert(d_sum_weight_grad);
		assert(weight_size > 0);
#endif
		sum_weight_grad.resize(weight_size);
		CUDA_SAFE_CALL(cudaMemcpy(&sum_weight_grad[0], d_sum_weight_grad, weight_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetSumBiasGrad(std::vector <REAL> &sum_bias_grad)
	{
#ifndef NO_CHECK
		assert(d_sum_bias_grad);
		assert(bias_size > 0);
#endif
		sum_bias_grad.resize(bias_size);
		CUDA_SAFE_CALL(cudaMemcpy(&sum_bias_grad[0], d_sum_bias_grad, bias_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetGrad(std::vector <REAL> &grad)
	{
#ifndef NO_CHECK
		assert(d_grad);
		assert(grad_size > 0);
#endif
		grad.resize(grad_size);
		CUDA_SAFE_CALL(cudaMemcpy(&grad[0], d_grad, grad_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetMask(std::vector <REAL> &mask)
	{
#ifndef NO_CHECK
		assert(d_grad);
		assert(mask_size > 0);
#endif
		mask.resize(mask_size);
		CUDA_SAFE_CALL(cudaMemcpy(&mask[0], d_mask, mask_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::GetDeriv(std::vector <REAL> &deriv)
	{
#ifndef NO_CHECK
		assert(d_out_deriv);
		assert(out_total_size > 0);
#endif
		deriv.resize(out_total_size);
		if(d_out_deriv == NULL)
		{
			return;
		}
		if(out_total_size == 0)
		{
			return;
		}
		CUDA_SAFE_CALL(cudaMemcpy(&deriv[0], d_out_deriv, out_total_size*sizeof(REAL), cudaMemcpyDeviceToHost));
	}

	void Layer::ZeroOutput()
	{
		CUDA_SAFE_CALL(cudaMemset(d_out, 0, out_total_size*sizeof(REAL)));
	}

	void Layer::ZeroSumGrad()
	{
		CUDA_SAFE_CALL(cudaMemset(d_sum_weight_grad, 0, weight_size*num_images*sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMemset(d_sum_bias_grad, 0, bias_size*num_images*sizeof(REAL)));
	}

	void Layer::ZeroMaxPool()
	{
		CUDA_SAFE_CALL(cudaMemset(d_mask, 0, mask_size*sizeof(REAL)));
	}

	void Layer::SetWeightValue(int idx, REAL v)
	{
		//assert(row >= 0 && col >= 0 && row < weight_rows && col < weight_cols);
		assert(idx >= 0 && idx < weight_size);
		DeviceSetValue<<<1,1>>>(d_weights, idx, v);
	}

	REAL Layer::GetWeightValue(int idx)
	{
		//assert(row >= 0 && col >= 0 && row < weight_rows && col < weight_cols);
		assert(idx >= 0 && idx < weight_size);
		DeviceGetValue<<<1,1>>>(d_weights, idx, d_tmp);
		REAL ret;
		CUDA_SAFE_CALL(cudaMemcpy(&ret, d_tmp, sizeof(REAL), cudaMemcpyDeviceToHost));
		return ret;
	}

	void Layer::SetBiasValue(int idx, REAL v)
	{
		assert(idx >= 0 && idx < bias_size);
		DeviceSetValue<<<1,1>>>(d_biases, idx, v);
	}

	REAL Layer::GetBiasValue(int idx)
	{
		assert(idx >= 0 && idx < bias_size);
		DeviceGetValue<<<1,1>>>(d_biases, idx, d_tmp);
		REAL ret;
		CUDA_SAFE_CALL(cudaMemcpy(&ret, d_tmp, sizeof(REAL), cudaMemcpyDeviceToHost));
		return ret;
	}

	void Layer::RunSumGrads()
	{
		int blocks = ceil((REAL)weight_size/256);
		SumGrads<<<blocks, 256>>>(d_sum_weight_grad, weight_size, num_images, d_sum_weight_grad);
		blocks = ceil((REAL)bias_size/256);
		SumGrads<<<blocks, 256>>>(d_sum_bias_grad, bias_size, num_images, d_sum_bias_grad);
	}

	void Layer::CheckValues()
	{
		vector <REAL> data;
		GetWeights(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetMomentumWeights(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetBiases(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetMomentumBiases(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetOutput(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetSumWeightGrad(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetSumBiasGrad(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetGrad(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetMask(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
		GetDeriv(data);
		for(size_t i=0; i < data.size(); i++)
		{
			assert(isfinite(data[i]));
		}
	}

} // namespace
