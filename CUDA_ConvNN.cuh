#ifndef __CUDA_CONVNN_CUH__
#define __CUDA_CONVNN_CUH__

// Moved CUDA kernels out of .cu files
#include "CUDA_Common.h"

using namespace std;

namespace CNN
{

	__global__ void Convolve(int offset, LayerType type, const REAL *images, int image_size, int channels,
	                         const REAL *conv_kernels, int kernel_size, int out_channels,
	                         const REAL *bias,
	                         REAL *output, REAL *output_deriv);

	template <int IN_SIZE>
	__global__ void ConvolveFast(int offset, LayerType type, const REAL *images, int in_width, int in_channels,
	                             const REAL *conv_kernels, int conv_size, int out_channels,
	                             const REAL *bias,
	                             REAL *output, REAL *output_deriv);

	__global__ void MaxPool(const REAL *images, int image_size, int channels, REAL *output, REAL *mask);
	__global__ void AvgPool(const REAL *images, int image_size, int channels, REAL *output, REAL *mask);

	__global__ void NN(int offset, LayerType type, const REAL *input, int input_size,
	                   const REAL *weights, int out_neurons, const REAL *bias,
	                   REAL *output, REAL *output_deriv);

	__global__ void SoftMax(const REAL *input, int num_input, REAL *output);

	// Backprop stuff
	__global__ void Y_minus_target(const REAL *input, const REAL *target, int target_size, REAL *output);

	__global__ void BackpropNN(int offset,
	                           const REAL *cur_layer_grad,
	                           const REAL *input, int input_size,
	                           const REAL *output_deriv, int output_size,
	                           const REAL *layer_weights, int weight_rows, int weight_cols,
	                           REAL *sum_weight_grad, REAL *sum_bias_grad);

	__global__ void BackpropNN2(int thread_offset,
	                            const REAL *cur_layer_grad,
	                            const REAL *input, int input_size,
	                            const REAL *output_deriv, int output_size,
	                            const REAL *layer_weights, int weight_rows, int weight_cols,
	                            REAL *prev_layer_grad);

	__global__ void BackpropMaxPool(const REAL *layer_grad, int out_size, int channels, const REAL *mask, REAL *prev_layer_grad);
	__global__ void BackpropAvgPool(const REAL *layer_grad, int out_size, int channels, const REAL *mask, REAL *prev_layer_grad);

	__global__ void BackpropConv(const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                             const REAL *input, int in_size, int input_channels,
	                             int output_channels, int conv_size,
	                             REAL *sum_weight_grad, REAL *sum_bias_grad);

	template <int IN_SIZE>
	__global__ void BackpropConvFast(int in_idx, int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                                 const REAL *input, int in_size, int input_channels,
	                                 int output_channels, int conv_size,
	                                 REAL *sum_weight_grad, REAL *sum_bias_grad);

	__global__ void BackpropConv2(int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                              const REAL *input, int in_size, int input_channels,
	                              int output_channels, int conv_size,
	                              REAL *prev_layer_grad);

	template <int OUT_SIZE>
	__global__ void BackpropConv2Fast(int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                                  const REAL *input, int in_size, int input_channels,
	                                  int output_channels, int conv_size,
	                                  REAL *prev_layer_grad);

	__global__ void AvgGrads(const REAL *weight_grad, int weight_size, int num_images, REAL *avg_weight);
	__global__ void UpdateWeights(REAL *momentum_delta, const REAL *delta, int weight_size, REAL learning_rate, REAL momentum_rate, REAL *weights);



	// each block = 1 image
	__global__ void Convolve(int offset, LayerType type, const REAL *images, int image_size, int in_channels,
	                         const REAL *conv_kernels, int kernel_size, int out_channels,
	                         const REAL *bias,
	                         REAL *output, REAL *output_deriv)
	{
		int image_idx = blockIdx.x;
		int idx = threadIdx.x + offset;
		int image_size2 = image_size - kernel_size + 1;
		if(idx >= image_size2*image_size2*out_channels)
		{
			return;
		}
		int out_idx = idx/(image_size2*image_size2);
		idx = idx - out_idx*image_size2*image_size2;
		int y = idx / image_size2;
		int x = idx - y*image_size2;
		// Set pointers to the image index
		const REAL *in  = &images[image_idx * image_size * image_size * in_channels];
		REAL *out       = &output[image_idx * image_size2 * image_size2 * out_channels];
		REAL *out_deriv = &output_deriv[image_idx * image_size2 * image_size2 * out_channels];
		REAL *out2       = &out[out_idx * image_size2 * image_size2];
		REAL *out_deriv2 = &out_deriv[out_idx * image_size2 * image_size2];
		REAL sum = bias[out_idx];
		for(int in_idx=0; in_idx < in_channels; in_idx++)
		{
			const REAL *in2    = &in[in_idx * image_size * image_size];
			const REAL *kernel = &conv_kernels[(in_idx*out_channels + out_idx)*kernel_size*kernel_size];
			for(int i=0; i < kernel_size; i++)
			{
				for(int j=0; j < kernel_size; j++)
				{
					sum += in2[(y+i)*image_size + x+j] * kernel[i*kernel_size+j];
				}
			}
		}
		// Apply non-linear function, if any
		REAL deriv = 1.0; // linear
		REAL tmp;
		switch(type)
		{
			case CONV_RELU:
				tmp=exp(sum);
				deriv=tmp/(1.0+tmp);
				sum = log(1.0+tmp);
				break;
			case CONV_TANH:
				sum = tanh(sum);
				deriv = 1.0 - sum*sum;
				break;
			case CONV_ABS_TANH:
				sum = tanh(sum);
				if(sum < 0)
				{
					deriv = -(1.0 - sum*sum);
				}
				else
				{
					deriv = (1.0 - sum*sum);
				}
				sum = fabs(sum);
				break;
			case CONV_SIGMOID:
				sum = 1.0 / (1.0 + exp(-sum));
				deriv = sum * (1.0 - sum);
				break;
		}
		out2[y*image_size2 + x] = sum;
		out_deriv2[y*image_size2 + x] = deriv;
	}

	template <int IN_SIZE>
	__global__ void ConvolveFast(int offset, LayerType type, const REAL *images, int in_width, int in_channels,
	                             const REAL *conv_kernels, int conv_size, int out_channels,
	                             const REAL *bias,
	                             REAL *output, REAL *output_deriv)
	{
		int image_idx = blockIdx.x;
		int idx = threadIdx.x + offset;
		int out_width = in_width - conv_size + 1;
		// Pre-mature exit
		//    if(idx >= out_width*out_width*out_channels) {
		//        return;
		//    }
		// At this point the values derived below can be out of bound because the above check is omitted
		// This is okay, because no memory is access yet
		int out_idx = idx / (out_width*out_width);
		idx = idx - out_idx*out_width*out_width;
		int y = idx / out_width;
		int x = idx - y*out_width;
		// Re-init
		idx = threadIdx.x + offset;
		// Set pointers to the image index
		const REAL *in  = &images[image_idx * in_width * in_width * in_channels];
		REAL *out       = &output[image_idx * out_width * out_width * out_channels];
		REAL *out_deriv = &output_deriv[image_idx * out_width * out_width * out_channels];
		// Set pointers within the images
		REAL *out2       = &out[out_idx * out_width * out_width];
		REAL *out_deriv2 = &out_deriv[out_idx * out_width * out_width];
		REAL sum;
		if(idx < out_width*out_width*out_channels)
		{
			sum = bias[out_idx];
		}
		for(int in_idx=0; in_idx < in_channels; in_idx++)
		{
			const REAL *in2 = &in[in_idx * in_width * in_width];
			__shared__ REAL s_in2[IN_SIZE*IN_SIZE];
			// The double __syncthreads looks weird, but is required
			__syncthreads(); // wait for any PREVIOUS stuff to finish
			if(threadIdx.x < in_width*in_width)
			{
				s_in2[threadIdx.x] = in2[threadIdx.x];
				// ASSUMES num threads = 256
				if(IN_SIZE == 32)
				{
					s_in2[256 + threadIdx.x] = in2[256 + threadIdx.x];
					s_in2[512 + threadIdx.x] = in2[512 + threadIdx.x];
					s_in2[768 + threadIdx.x] = in2[768 + threadIdx.x];
				}
				else if(IN_SIZE == 24)
				{
					s_in2[256 + threadIdx.x] = in2[256 + threadIdx.x];
					if(threadIdx.x < 64)
					{
						s_in2[512 + threadIdx.x] = in2[512 + threadIdx.x];
					}
				}
			}
			__syncthreads();
			// Looks weird hey
			if(idx >= out_width*out_width*out_channels)
			{
				continue;
			}
			const REAL *kernel = &conv_kernels[(in_idx*out_channels + out_idx)*conv_size*conv_size];
			for(int i=0; i < conv_size; i++)
			{
				for(int j=0; j < conv_size; j++)
				{
					int in_idx = (y+i)*in_width + x+j;
					sum += s_in2[in_idx] * kernel[i*conv_size+j];
				}
			}
		}
		if(idx >= out_width*out_width*out_channels)
		{
			return;
		}
		// Apply non-linear function, if any
		REAL deriv = 1; // linear
		switch(type)
		{
			case CONV_RELU:
				sum = max(0.0f, sum);
				if(sum == 0)
				{
					deriv = 0;
				}
				break;
			case CONV_TANH:
				sum = tanh(sum);
				deriv = 1 - sum*sum;
				break;
			case CONV_ABS_TANH:
				sum = tanh(sum);
				if(sum < 0)
				{
					deriv = -(1 - sum*sum);
				}
				else
				{
					deriv = (1 - sum*sum);
				}
				sum = fabs(sum);
				break;
			case CONV_SIGMOID:
				sum = 1 / (1 + exp(-sum));
				deriv = sum * (1 - sum);
				break;
		}
		out2[y*out_width + x] = sum;
		out_deriv2[y*out_width + x] = deriv;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// NOTE: due to thread limit, each threadIdx.x = x
	__global__ void MaxPool(int offset, const REAL *images, int image_size, int channels, REAL *output, REAL *mask)
	{
		int image_idx = blockIdx.x;
		int idx = threadIdx.x + offset;
		int image_size2 = image_size/2;
		if(idx >= image_size2*image_size2*channels)
		{
			return;
		}
		int ch = idx / (image_size2*image_size2);
		idx = idx - ch*image_size2*image_size2;
		int y = idx / image_size2;
		int x = idx - y*image_size2;
		const REAL *in = &images[image_idx * image_size * image_size * channels];
		REAL *mask2    = &mask[image_idx * image_size * image_size * channels];
		REAL *out      = &output[image_idx * image_size2 * image_size2 * channels];
		// for(int c=0; c < channels; c++) {
		const REAL *in2 = &in[ch * image_size * image_size];
		REAL *out2      = &out[ch * image_size2 * image_size2];
		REAL *m2        = &mask2[ch * image_size * image_size];
		// for(int y=0; y < image_size2; y++) {
		REAL a = in2[(y*2)*image_size + x*2];
		REAL b = in2[(y*2)*image_size + x*2+1];
		REAL c = in2[(y*2+1)*image_size + x*2];
		REAL d = in2[(y*2+1)*image_size + x*2+1];
		REAL m = max(a, max(b, max(c, d)));
		if(a == m)
		{
			out2[y*image_size2 + x] = a;
			m2[(y*2)*image_size + x*2] = 1;
		}
		else if(b == m)
		{
			out2[y*image_size2 + x] = b;
			m2[(y*2)*image_size + x*2+1] = 1;
		}
		else if(c == m)
		{
			out2[y*image_size2 + x] = c;
			m2[(y*2+1)*image_size + x*2] = 1;
		}
		else
		{
			out2[y*image_size2 + x] = d;
			m2[(y*2+1)*image_size + x*2+1] = 1;
		}
		// }
		// }
	}

	__global__ void AvgPool(int offset, const REAL *images, int image_size, int channels, REAL *output, REAL *mask)
	{
		int image_idx = blockIdx.x;
		int idx = threadIdx.x + offset;
		int image_size2 = image_size/2;
		if(idx >= image_size2*image_size2*channels)
		{
			return;
		}
		int ch = idx / (image_size2*image_size2);
		idx = idx - ch*image_size2*image_size2;
		int y = idx / image_size2;
		int x = idx - y*image_size2;
		const REAL *in = &images[image_idx * image_size * image_size * channels];
		REAL *mask2    = &mask[image_idx * image_size * image_size * channels];
		REAL *out      = &output[image_idx * image_size2 * image_size2 * channels];
		// for(int c=0; c < channels; c++) {
		const REAL *in2 = &in[ch * image_size * image_size];
		REAL *out2      = &out[ch * image_size2 * image_size2];
		REAL *m2        = &mask2[ch * image_size * image_size];
		// for(int y=0; y < image_size2; y++) {
		REAL a = in2[(y*2)*image_size + x*2];
		REAL b = in2[(y*2)*image_size + x*2+1];
		REAL c = in2[(y*2+1)*image_size + x*2];
		REAL d = in2[(y*2+1)*image_size + x*2+1];
		out2[y*image_size2 + x] = (a+b+c+d)*0.25;
		m2[(y*2)*image_size + x*2] = 0.25;
		m2[(y*2)*image_size + x*2+1] = 0.25;
		m2[(y*2+1)*image_size + x*2] = 0.25;
		m2[(y*2+1)*image_size + x*2+1] = 0.25;
		// }
		// }
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void NN(int offset, LayerType type, const REAL *input, int input_size,
	                   const REAL *weights, int out_neurons, const REAL *bias,
	                   REAL *output, REAL *output_deriv)
	{
		// thread on output index
		int image_idx = blockIdx.x;
		int out_idx = threadIdx.x + offset;
		if(out_idx >= out_neurons)
		{
			return;
		}
		const REAL *in  = &input[image_idx * input_size];
		REAL *out       = &output[image_idx * out_neurons];
		REAL *out_deriv = &output_deriv[image_idx * out_neurons];
		// Slow method for now
		const REAL *row = &weights[out_idx*input_size];
		REAL sum = bias[out_idx];
		for(int n=0; n < input_size; n++)
		{
			sum += row[n] * in[n];
		}
		REAL deriv = 1;
		switch(type)
		{
			case NN_RELU:
				sum = max(0.0f, sum);
				if(sum == 0)
				{
					deriv = 0;
				}
				break;
			case NN_TANH:
				sum = tanh(sum);
				deriv = 1 - sum*sum;
				break;
			case NN_ABS_TANH:
				sum = tanh(sum);
				if(sum < 0)
				{
					deriv = -(1 - sum*sum);
				}
				else
				{
					deriv = (1 - sum*sum);
				}
				sum = fabs(sum);
				break;
			case NN_SIGMOID:
				sum = 1 / (1 + exp(-sum));
				deriv = sum * (1 - sum);
				break;
		}
		out[out_idx] = sum;
		out_deriv[out_idx] = deriv;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void SoftMax(const REAL *input, int num_input, REAL *output)
	{
		int image_idx = blockIdx.x;
		const REAL *in = &input[image_idx * num_input];
		REAL       *out = &output[image_idx * num_input];
		// numerically stable version
		REAL big = -FLT_MAX;
		for(int i=0; i < num_input; i++)
		{
			big = max(big, in[i]);
		}
		REAL sum = 0;
		for(int i=0; i < num_input; i++)
		{
			sum += exp(in[i] - big);
		}
		REAL log_norm = log(sum) + big;
		for(int i=0; i < num_input; i++)
		{
			out[i] = exp(in[i] - log_norm);
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void Y_minus_target(const REAL *Y, const REAL *target, int target_size, REAL *output)
	{
		int image_idx = blockIdx.x;
		int i = threadIdx.x;
		int idx = image_idx * target_size;
		const REAL *y = &Y[idx];
		const REAL *t = &target[idx];
		REAL *out     = &output[idx];
		out[i] = y[i] - t[i];
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void BackpropMaxPool(const REAL *layer_grad, int out_size, int channels, const REAL *mask, REAL *prev_layer_grad)
	{
		// number of threads = output image width
		int image_idx = blockIdx.x;
		int x = threadIdx.x;
		int in_size = out_size*2;
		const REAL *cur_grad = &layer_grad[image_idx * out_size * out_size * channels];
		const REAL *m        = &mask[image_idx * in_size * in_size * channels];
		REAL *prev_grad      = &prev_layer_grad[image_idx * in_size * in_size * channels];
		for(int c=0; c < channels; c++)
		{
			const REAL *cur_grad2 = &cur_grad[c * out_size * out_size];
			const REAL *m2        = &m[c * in_size * in_size];
			REAL *prev_grad2      = &prev_grad[c * in_size * in_size];
			for(int y=0; y < out_size; y++)
			{
				REAL g = cur_grad2[y*out_size + x];
				prev_grad2[y*2*in_size + x*2]       = g*m2[y*2*in_size + x*2];
				prev_grad2[y*2*in_size + x*2+1]     = g*m2[y*2*in_size + x*2+1];
				prev_grad2[(y*2+1)*in_size + x*2]   = g*m2[(y*2+1)*in_size + x*2];
				prev_grad2[(y*2+1)*in_size + x*2+1] = g*m2[(y*2+1)*in_size + x*2+1];
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void BackpropAvgPool(const REAL *layer_grad, int out_size, int channels, const REAL *mask, REAL *prev_layer_grad)
	{
		// number of threads = output image width
		int image_idx = blockIdx.x;
		int x = threadIdx.x;
		int in_size = out_size*2;
		const REAL *cur_grad = &layer_grad[image_idx * out_size * out_size * channels];
		const REAL *m        = &mask[image_idx * in_size * in_size * channels];
		REAL *prev_grad      = &prev_layer_grad[image_idx * in_size * in_size * channels];
		for(int c=0; c < channels; c++)
		{
			const REAL *cur_grad2 = &cur_grad[c * out_size * out_size];
			const REAL *m2        = &m[c * in_size * in_size];
			REAL *prev_grad2      = &prev_grad[c * in_size * in_size];
			for(int y=0; y < out_size; y++)
			{
				REAL g = cur_grad2[y*out_size + x];
				prev_grad2[y*2*in_size + x*2]       = g*m2[y*2*in_size + x*2];
				prev_grad2[y*2*in_size + x*2+1]     = g*m2[y*2*in_size + x*2+1];
				prev_grad2[(y*2+1)*in_size + x*2]   = g*m2[(y*2+1)*in_size + x*2];
				prev_grad2[(y*2+1)*in_size + x*2+1] = g*m2[(y*2+1)*in_size + x*2+1];
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void BackpropConv(int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                             const REAL *input, int in_size, int input_channels,
	                             int output_channels, int conv_size,
	                             REAL *sum_weight_grad, REAL *sum_bias_grad)
	{
		// Pull-through version
		int image_idx = blockIdx.x;
		int out_size = in_size - conv_size + 1;
		int idx = threadIdx.x + offset;
		if(idx >= input_channels*output_channels*conv_size*conv_size)
		{
			return;
		}
		int in_idx = idx / (output_channels*conv_size*conv_size);
		idx = idx - in_idx*output_channels*conv_size*conv_size;
		int out_idx = idx / (conv_size*conv_size);
		idx = idx - out_idx*conv_size*conv_size;
		int a = idx/conv_size;
		int b = idx - a*conv_size;
		// Set pointers to image index
		const REAL *in    = &input[image_idx * in_size * in_size * input_channels];
		const REAL *grad  = &layer_grad[image_idx * out_size * out_size * output_channels];
		const REAL *deriv = &layer_deriv[image_idx * out_size * out_size * output_channels];
		REAL *weight_grad = &sum_weight_grad[image_idx * conv_size * conv_size * input_channels * output_channels];
		REAL *bias_grad   = &sum_bias_grad[image_idx * input_channels * output_channels];
		// Set pointers to within image index
		const REAL *in2    = &in[in_idx * in_size * in_size];
		const REAL *grad2  = &grad[out_idx * out_size * out_size];
		const REAL *deriv2 = &deriv[out_idx * out_size * out_size];
		REAL *weight_grad2 = &weight_grad[(in_idx*output_channels + out_idx)*conv_size*conv_size];
		REAL *bias_grad2   = &bias_grad[in_idx*output_channels + out_idx];
		REAL sum_bias=0;
		REAL sum_weight=0;
		int k=0;
		for(int y=0; y < out_size; y++)
		{
			for(int x=0; x < out_size; x++)
			{
				REAL v = grad2[k] * deriv2[k];
				sum_bias += v;
				// Performance killer right here
				sum_weight += in2[(y+a)*in_size + (x+b)] * v;
				k++;
			}
		}
		*bias_grad2 = sum_bias;
		weight_grad2[a*conv_size + b] = sum_weight;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <int IN_SIZE>
	__global__ void BackpropConvFast(int in_idx, int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                                 const REAL *input, int in_size, int input_channels,
	                                 int output_channels, int conv_size,
	                                 REAL *sum_weight_grad, REAL *sum_bias_grad)
	{
		// Pull-through version
		int image_idx = blockIdx.x;
		int out_size = in_size - conv_size + 1;
		int idx = threadIdx.x + offset;
		const REAL *in  = &input[image_idx * in_size * in_size * input_channels];
		const REAL *in2 = &in[in_idx * in_size * in_size];
		__shared__ REAL s_in2[IN_SIZE*IN_SIZE];
		if(threadIdx.x < in_size*in_size)
		{
			s_in2[threadIdx.x] = in2[threadIdx.x];
			if(IN_SIZE == 32)
			{
				s_in2[256 + threadIdx.x] = in2[256 + threadIdx.x];
				s_in2[512 + threadIdx.x] = in2[512 + threadIdx.x];
				s_in2[768 + threadIdx.x] = in2[768 + threadIdx.x];
			}
			else if(IN_SIZE == 24)
			{
				s_in2[256 + threadIdx.x] = in2[256 + threadIdx.x];
				if(threadIdx.x < 64)
				{
					s_in2[512 + threadIdx.x] = in2[512 + threadIdx.x];
				}
			}
		}
		__syncthreads();
		if(idx >= output_channels*conv_size*conv_size)
		{
			return;
		}
		int out_idx = idx / (conv_size*conv_size);
		idx = idx - out_idx*(conv_size*conv_size);
		int a = idx / conv_size;
		int b = idx - a*conv_size;
		// Set pointers to image index
		const REAL *grad  = &layer_grad[image_idx * out_size * out_size * output_channels];
		const REAL *deriv = &layer_deriv[image_idx * out_size * out_size * output_channels];
		REAL *weight_grad = &sum_weight_grad[image_idx * conv_size * conv_size * input_channels * output_channels];
		REAL *bias_grad   = &sum_bias_grad[image_idx * input_channels * output_channels];
		// Set pointers to within image index
		const REAL *grad2  = &grad[out_idx * out_size * out_size];
		const REAL *deriv2 = &deriv[out_idx * out_size * out_size];
		REAL *weight_grad2 = &weight_grad[(in_idx*output_channels + out_idx)*conv_size*conv_size];
		REAL *bias_grad2   = &bias_grad[in_idx*output_channels + out_idx];
		REAL sum_bias=0;
		REAL sum_weight=0;
		int k=0;
		for(int y=0; y < out_size; y++)
		{
			for(int x=0; x < out_size; x++)
			{
				REAL v = grad2[k] * deriv2[k];
				sum_bias += v;
				// Performance killer right here
				sum_weight += s_in2[(y+a)*in_size + (x+b)] * v;
				k++;
			}
		}
		*bias_grad2 = sum_bias;
		weight_grad2[a*conv_size + b] = sum_weight;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void BackpropConv2(int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                              const REAL *input, int in_size, int input_channels,
	                              int output_channels, int conv_size,
	                              REAL *prev_layer_grad)
	{
		// Pull-through version for prev layer grad only
		int image_idx = blockIdx.x;
		int idx = threadIdx.x + offset;
		if(idx >= input_channels*in_size*in_size)
		{
			return;
		}
		int in_idx = idx / (in_size*in_size);
		idx = idx - in_idx*in_size*in_size;
		int y = idx / in_size;
		int x = idx - y*in_size;
		int out_size = in_size - conv_size + 1;
		const REAL *grad  = &layer_grad[image_idx * out_size * out_size * output_channels];
		const REAL *deriv = &layer_deriv[image_idx * out_size * out_size * output_channels];
		REAL *prev_grad   = &prev_layer_grad[image_idx * in_size * in_size * input_channels];
		REAL *prev_grad2  = &prev_grad[in_idx * in_size * in_size];
		REAL sum=0;
		for(int out_idx=0; out_idx < output_channels; out_idx++)
		{
			const REAL *grad2  = &grad[out_idx * out_size * out_size];
			const REAL *deriv2 = &deriv[out_idx * out_size * out_size];
			const REAL *kernel = &conv_kernel[(in_idx*output_channels + out_idx)*conv_size*conv_size];
			for(int a=0; a < conv_size; a++)
			{
				for(int b=0; b < conv_size; b++)
				{
					int yy = y - a;
					int xx = x - b;
					if(xx < 0 || xx >= out_size|| yy < 0 || yy >= out_size)
					{
						continue;
					}
					REAL w = kernel[a*conv_size + b];
					REAL d = deriv2[yy*out_size + xx];
					REAL g = grad2[yy*out_size + xx];
					sum += w*(g*d);
				}
			}
		}
		prev_grad2[y*in_size + x] = sum;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <int OUT_SIZE>
	__global__ void BackpropConv2Fast(int offset, const REAL *layer_grad, const REAL *layer_deriv, const REAL *conv_kernel,
	                                  const REAL *input, int in_size, int input_channels,
	                                  int output_channels, int conv_size,
	                                  REAL *prev_layer_grad)
	{
		// Pull-through version for prev layer grad only
		int image_idx = blockIdx.x;
		int idx = threadIdx.x + offset;
		//    if(idx >= input_channels*in_size*in_size) {
		//        return;
		//    }
		int in_idx = idx / (in_size*in_size);
		idx = idx - in_idx*in_size*in_size;
		int y = idx / in_size;
		int x = idx - y*in_size;
		// Reset
		idx = threadIdx.x + offset;
		int out_size = in_size - conv_size + 1;
		const REAL *grad  = &layer_grad[image_idx * out_size * out_size * output_channels];
		const REAL *deriv = &layer_deriv[image_idx * out_size * out_size * output_channels];
		REAL *prev_grad   = &prev_layer_grad[image_idx * in_size * in_size * input_channels];
		REAL *prev_grad2  = &prev_grad[in_idx * in_size * in_size];
		REAL sum=0;
		for(int out_idx=0; out_idx < output_channels; out_idx++)
		{
			const REAL *grad2  = &grad[out_idx * out_size * out_size];
			const REAL *deriv2 = &deriv[out_idx * out_size * out_size];
			const REAL *kernel = &conv_kernel[(in_idx*output_channels + out_idx)*conv_size*conv_size];
			// Cache here
			__shared__ REAL s_grad_deriv[OUT_SIZE*OUT_SIZE];
			if(threadIdx.x < out_size*out_size)
			{
				s_grad_deriv[threadIdx.x] = grad2[threadIdx.x] * deriv2[threadIdx.x];
			}
			__syncthreads();
			if(idx < input_channels*in_size*in_size)
			{
				for(int a=0; a < conv_size; a++)
				{
					for(int b=0; b < conv_size; b++)
					{
						int yy = y - a;
						int xx = x - b;
						if(xx < 0 || xx >= out_size|| yy < 0 || yy >= out_size)
						{
							continue;
						}
						REAL w = kernel[a*conv_size + b];
						sum += w * s_grad_deriv[yy*out_size + xx];
					}
				}
			}
			__syncthreads();
		}
		if(idx >= input_channels*in_size*in_size)
		{
			return;
		}
		prev_grad2[y*in_size + x] = sum;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void BackpropNN(int offset,
	                           const REAL *cur_layer_grad,
	                           const REAL *input, int input_size,
	                           const REAL *output_deriv, int output_size,
	                           const REAL *layer_weights, int weight_rows, int weight_cols,
	                           REAL *sum_weight_grad, REAL *sum_bias_grad)
	{
		int image_idx = blockIdx.x;
		int out_idx = threadIdx.x + offset;
		if(out_idx >= output_size)
		{
			return;
		}
		const REAL *grad      = &cur_layer_grad[image_idx * output_size];
		const REAL *in        = &input[image_idx * input_size];
		const REAL *out_deriv = &output_deriv[image_idx * output_size];
		REAL *weight_grad     = &sum_weight_grad[image_idx * weight_rows * weight_cols];
		REAL *bias_grad       = &sum_bias_grad[image_idx * output_size];
		REAL g = grad[out_idx] * out_deriv[out_idx];
		// outer product
		for(int j=0; j < input_size; j++)
		{
			weight_grad[out_idx*input_size + j] = in[j] * g;
		}
		bias_grad[out_idx] = g;
	}

	__global__ void BackpropNN2(int thread_offset,
	                            const REAL *cur_layer_grad,
	                            const REAL *input, int input_size,
	                            const REAL *output_deriv, int output_size,
	                            const REAL *layer_weights, int weight_rows, int weight_cols,
	                            REAL *prev_layer_grad)
	{
		int image_idx = blockIdx.x;
		int j =  threadIdx.x + thread_offset;
		if(j >= weight_cols)
		{
			return;
		}
		const REAL *grad      = &cur_layer_grad[image_idx * output_size];
		const REAL *out_deriv = &output_deriv[image_idx * output_size];
		REAL *prev_grad       = &prev_layer_grad[image_idx * weight_cols];
		// size of the input
		//for(int j=0; j < weight_cols; j++) {
		REAL sum=0;
		for(int i=0; i < weight_rows; i++)
		{
			sum += layer_weights[i*weight_cols + j] * grad[i] * out_deriv[i];
		}
		prev_grad[j] = sum;
		//}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void AvgGrads(const REAL *weight_grad, int weight_size, int num_images, REAL *avg_weight)
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
		avg_weight[idx] = sum / num_images;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void UpdateWeights(REAL *momentum_delta, const REAL *delta, int weight_size, REAL learning_rate, REAL momentum_rate, REAL *weights)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if(idx >= weight_size)
		{
			return;
		}
		momentum_delta[idx] = momentum_rate*momentum_delta[idx] + learning_rate*delta[idx];
		weights[idx] = weights[idx] - momentum_delta[idx];
	}

} // namespace
#endif
