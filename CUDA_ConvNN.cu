#include <cstdio>
#include <float.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>

#include "CUDA_ConvNN.h"
#include "CUDA_ConvNN.cuh"
#include "CUDA_ConvNN_Layer.h"
#include "CUDA_Common.h"

using namespace std;

//#define TIMING

bool g_use_fast = true;

namespace CNN
{

	CudaConvNN::CudaConvNN()
	{
		m_momentum_init = false;
		m_batch_size = 0;
		m_image_width = 0;
		m_image_channels = 0;
		d_images = NULL;
		d_labels = NULL;
		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);
	}

	CudaConvNN::~CudaConvNN()
	{
		if(d_images)
		{
			cudaFree(d_images);
		}
		if(d_labels)
		{
			cudaFree(d_labels);
		}
		for(size_t i=0; i < m_layers.size(); i++)
		{
			m_layers[i].CleanUp();
		}
	}

	void CudaConvNN::Init(int image_width, int channels, int categories, int batch_size)
	{
		assert(m_layers.empty());
		assert(image_width > 0);
		assert(channels > 0);
		assert(categories > 0);
		assert(batch_size > 0);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_images, batch_size * image_width * image_width * channels * sizeof(REAL)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_labels, batch_size * categories * sizeof(REAL)));
		m_batch_size = batch_size;
		m_image_width = image_width;
		m_image_channels = channels;
	}

	void CudaConvNN::AddLayer(LayerType type, int conv_size, int out_channels)
	{
		assert(type == CONV_RELU || type == CONV_TANH || type == CONV_ABS_TANH || type == CONV_SIGMOID);
		assert(d_images);
		assert(d_labels);
		Layer L;
		if(m_layers.empty())
		{
			L.num_images = m_batch_size;
			L.d_in = d_images;
			L.in_image_width = m_image_width;
			L.in_channels = m_image_channels;
		}
		else
		{
			L.num_images = m_layers.back().num_images;
			L.d_in = m_layers.back().d_out;
			L.in_image_width = m_layers.back().out_image_width;
			L.in_channels = m_layers.back().out_channels;
		}
		L.conv_size = conv_size;
		L.out_channels = out_channels;
		L.InitConv(type);
		L.InitRandWeights(0.1);
		m_layers.push_back(L);
	}

	void CudaConvNN::AddLayer(LayerType type, int out_size)
	{
		assert(type == NN_LINEAR || type == NN_RELU || type == NN_TANH || type == NN_ABS_TANH || type == NN_SIGMOID);
		assert(d_images);
		assert(d_labels);
		Layer L;
		if(m_layers.empty())
		{
			L.num_images = m_batch_size;
			L.d_in = d_images;
			L.in_size = m_image_width * m_image_width * m_image_channels;
		}
		else
		{
			L.num_images = m_layers.back().num_images;
			L.d_in = m_layers.back().d_out;
			if(m_layers.back().out_size)
			{
				L.in_size = m_layers.back().out_size;
			}
			else
			{
				L.in_size = m_layers.back().out_size_per_sample;
			}
		}
		L.out_size = out_size;
		L.InitNN(type);
		L.InitRandWeights(0.1);
		m_layers.push_back(L);
	}

	void CudaConvNN::AddLayer(LayerType type)
	{
		assert(type == MAX_POOL || type == AVG_POOL || type == NN_SOFTMAX);
		assert(d_images);
		assert(d_labels);
		assert(m_image_width > 0);
		assert(m_image_channels > 0);
		assert(m_batch_size > 0);
		Layer L;
		if(m_layers.empty())
		{
			L.num_images = m_batch_size;
			L.d_in = d_images;
			L.in_image_width = m_image_width;
			L.in_channels = m_image_channels;
			L.in_size = m_image_width * m_image_width * m_image_channels;
		}
		else
		{
			L.num_images = m_layers.back().num_images;
			L.d_in = m_layers.back().d_out;
			L.in_image_width = m_layers.back().out_image_width;
			L.in_channels = m_layers.back().out_channels;
			L.in_size = m_layers.back().out_size;
		}
		if(type == MAX_POOL || type == AVG_POOL)
		{
			L.InitPool(type);
		}
		else if(type == NN_SOFTMAX)
		{
			L.InitSoftMax();
		}
		else
		{
			assert(0);
		}
		m_layers.push_back(L);
	}

	void CudaConvNN::FeedForward()
	{
		// Forward pass
		for(size_t l=0; l < m_layers.size(); l++)
		{
			Layer &L = m_layers[l];
			L.ZeroOutput();
			L.ZeroSumGrad();
			L.ZeroMaxPool();
#ifdef TIMING
			cudaEventRecord(m_start, 0);
			REAL time;
#endif
			if(L.type == CONV_RELU || L.type == CONV_LINEAR || L.type == CONV_TANH || L.type == CONV_ABS_TANH || L.type == CONV_SIGMOID)
			{
				if(g_use_fast && L.in_image_width == 32)
				{
					// assert(L.out_image_width*L.out_image_width < NUM_THREADS);
#ifdef TIMING
					printf("FAST CONVOLVE - 32\n");
#endif
					int iter = ceil((REAL)L.out_image_width*L.out_image_width*L.out_channels / NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int offset = i*NUM_THREADS;
						ConvolveFast<32><<<m_batch_size, NUM_THREADS>>>(offset, L.type, L.d_in, L.in_image_width ,L.in_channels,
						        L.d_weights, L.conv_size, L.out_channels, L.d_biases,
						        L.d_out, L.d_out_deriv);
					}
				}
				else if(g_use_fast && L.in_image_width == 24)
				{
					// assert(L.out_image_width*L.out_image_width < NUM_THREADS);
#ifdef TIMING
					printf("FAST CONVOLVE - 32\n");
#endif
					int iter = ceil((REAL)L.out_image_width*L.out_image_width*L.out_channels / NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int offset = i*NUM_THREADS;
						ConvolveFast<24><<<m_batch_size, NUM_THREADS>>>(offset, L.type, L.d_in, L.in_image_width ,L.in_channels,
						        L.d_weights, L.conv_size, L.out_channels, L.d_biases,
						        L.d_out, L.d_out_deriv);
					}
				}
				else if(g_use_fast && L.in_image_width == 14)
				{
#ifdef TIMING
					printf("FAST CONVOLVE - 14\n");
#endif
					assert(L.out_image_width*L.out_image_width < NUM_THREADS);
					int iter = ceil((REAL)L.out_image_width*L.out_image_width*L.out_channels / NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int offset = i*NUM_THREADS;
						ConvolveFast<14><<<m_batch_size, NUM_THREADS>>>(offset, L.type, L.d_in, L.in_image_width ,L.in_channels,
						        L.d_weights, L.conv_size, L.out_channels, L.d_biases,
						        L.d_out, L.d_out_deriv);
					}
				}
				else if(g_use_fast && L.in_image_width == 10)
				{
#ifdef TIMING
					printf("FAST CONVOLVE - 14\n");
#endif
					assert(L.out_image_width*L.out_image_width < NUM_THREADS);
					int iter = ceil((REAL)L.out_image_width*L.out_image_width*L.out_channels / NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int offset = i*NUM_THREADS;
						ConvolveFast<10><<<m_batch_size, NUM_THREADS>>>(offset, L.type, L.d_in, L.in_image_width ,L.in_channels,
						        L.d_weights, L.conv_size, L.out_channels, L.d_biases,
						        L.d_out, L.d_out_deriv);
					}
				}
				else
				{
					int iter = ceil((REAL)L.out_image_width*L.out_image_width*L.out_channels / NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int offset = i*NUM_THREADS;
						Convolve<<<m_batch_size, NUM_THREADS>>>(offset, L.type, L.d_in, L.in_image_width ,L.in_channels,
						                                        L.d_weights, L.conv_size, L.out_channels, L.d_biases,
						                                        L.d_out, L.d_out_deriv);
					}
				}
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Convolve: %f ms\n", l, time);
#endif
			}
			else if(L.type == MAX_POOL)
			{
				L.ZeroMaxPool();
				int iter = ceil((REAL)L.out_channels*L.out_image_width*L.out_image_width / NUM_THREADS);
				for(int i=0; i < iter; i++)
				{
					int offset = i*NUM_THREADS;
					MaxPool<<<m_batch_size, NUM_THREADS>>>(offset, L.d_in, L.in_image_width, L.in_channels, L.d_out, L.d_mask);
				}
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Max pool: %f ms\n", l, time);
#endif
			}
			else if(L.type == AVG_POOL)
			{
				int iter = ceil((REAL)L.out_channels*L.out_image_width*L.out_image_width / NUM_THREADS);
				for(int i=0; i < iter; i++)
				{
					int offset = i*NUM_THREADS;
					AvgPool<<<m_batch_size, NUM_THREADS>>>(offset, L.d_in, L.in_image_width, L.in_channels, L.d_out, L.d_mask);
				}
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Avg pool: %f ms\n", l, time);
#endif
			}
			else if(L.type == NN_LINEAR || L.type == NN_RELU || L.type == NN_TANH || L.type == NN_ABS_TANH || L.type == NN_SIGMOID)
			{
				int iter = ceil((REAL)L.out_size / NUM_THREADS);
				for(int i=0; i < iter; i++)
				{
					int offset = i*NUM_THREADS;
					NN<<<m_batch_size, NUM_THREADS>>>(offset, L.type, L.d_in, L.in_size,
					                                  L.d_weights, L.out_size, L.d_biases,
					                                  L.d_out, L.d_out_deriv);
				}
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, NN: %f ms\n", l, time);
#endif
			}
			else if(L.type == NN_SOFTMAX)
			{
				SoftMax<<<m_batch_size, 1>>>(L.d_in, L.in_size, L.d_out);
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, NN_SOFTMAX: %f ms\n", l, time);
#endif
			}
			else
			{
				printf("Layer not supported\n");
				assert(0);
			}
		}
	}

	size_t CudaConvNN::TotalMemUsed()
	{
		size_t sum=0;
		for(size_t i=0; i < m_layers.size(); i++)
		{
			sum += m_layers[i].total_mem_used;
		}
		return sum;
	}

	void CudaConvNN::BackProp(REAL learning_rate, REAL momentum_rate)
	{
		// Backward pass
		for(int l=(int)m_layers.size()-1; l >= 0; l--)
		{
			Layer &L = m_layers[l];
			REAL *prev_grad = NULL;
			if(l >= 1)
			{
				prev_grad = m_layers[l-1].d_grad;
			}
#ifdef TIMING
			cudaEventRecord(m_start, 0);
			REAL time;
#endif
			if(L.type == CONV_RELU || L.type == CONV_LINEAR || L.type == CONV_TANH || L.type == CONV_ABS_TANH || L.type == CONV_SIGMOID)
			{
				if(g_use_fast && L.in_image_width == 32)
				{
#ifdef TIMING
					printf("FAST BACKPROP - 32\n");
#endif
					// Faster version - if the image is small enough it can be cached fully in shared memory
					for(int in_idx=0; in_idx < L.in_channels; in_idx++)
					{
						int iter = ceil((REAL)L.out_channels*L.conv_size*L.conv_size / NUM_THREADS);
						for(int j=0; j < iter; j++)
						{
							int offset = j*NUM_THREADS;
							BackpropConvFast<32><<<m_batch_size, NUM_THREADS>>>
							(in_idx, offset, L.d_grad, L.d_out_deriv, L.d_weights,
							 L.d_in, L.in_image_width, L.in_channels,
							 L.out_channels, L.conv_size,
							 L.d_sum_weight_grad, L.d_sum_bias_grad);
						}
					}
				}
				else if(g_use_fast && L.in_image_width == 24)
				{
#ifdef TIMING
					printf("FAST BACKPROP - 32\n");
#endif
					// Faster version - if the image is small enough it can be cached fully in shared memory
					for(int in_idx=0; in_idx < L.in_channels; in_idx++)
					{
						int iter = ceil((REAL)L.out_channels*L.conv_size*L.conv_size / NUM_THREADS);
						for(int j=0; j < iter; j++)
						{
							int offset = j*NUM_THREADS;
							BackpropConvFast<24><<<m_batch_size, NUM_THREADS>>>
							(in_idx, offset, L.d_grad, L.d_out_deriv, L.d_weights,
							 L.d_in, L.in_image_width, L.in_channels,
							 L.out_channels, L.conv_size,
							 L.d_sum_weight_grad, L.d_sum_bias_grad);
						}
					}
				}
				else if(g_use_fast && L.in_image_width == 14)
				{
#ifdef TIMING
					printf("FAST BACKPROP - 14\n");
#endif
					// Faster version - if the image is small enough it can be cached fully in shared memory
					assert(L.in_image_width*L.in_image_width < NUM_THREADS);
					for(int in_idx=0; in_idx < L.in_channels; in_idx++)
					{
						int iter = ceil((REAL)L.out_channels*L.conv_size*L.conv_size / NUM_THREADS);
						for(int j=0; j < iter; j++)
						{
							int offset = j*NUM_THREADS;
							BackpropConvFast<14><<<m_batch_size, NUM_THREADS>>>
							(in_idx, offset, L.d_grad, L.d_out_deriv, L.d_weights,
							 L.d_in, L.in_image_width, L.in_channels,
							 L.out_channels, L.conv_size,
							 L.d_sum_weight_grad, L.d_sum_bias_grad);
						}
					}
				}
				else if(g_use_fast && L.in_image_width == 10)
				{
#ifdef TIMING
					printf("FAST BACKPROP - 14\n");
#endif
					// Faster version - if the image is small enough it can be cached fully in shared memory
					assert(L.in_image_width*L.in_image_width < NUM_THREADS);
					for(int in_idx=0; in_idx < L.in_channels; in_idx++)
					{
						int iter = ceil((REAL)L.out_channels*L.conv_size*L.conv_size / NUM_THREADS);
						for(int j=0; j < iter; j++)
						{
							int offset = j*NUM_THREADS;
							BackpropConvFast<10><<<m_batch_size, NUM_THREADS>>>
							(in_idx, offset, L.d_grad, L.d_out_deriv, L.d_weights,
							 L.d_in, L.in_image_width, L.in_channels,
							 L.out_channels, L.conv_size,
							 L.d_sum_weight_grad, L.d_sum_bias_grad);
						}
					}
				}
				else
				{
#ifdef TIMING
					printf("SLOW: %d\n", L.in_image_width);
#endif
					int iter = ceil((REAL)L.in_channels*L.out_channels*L.conv_size*L.conv_size / NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int offset = i*NUM_THREADS;
						BackpropConv<<<m_batch_size, NUM_THREADS>>>(offset, L.d_grad, L.d_out_deriv, L.d_weights,
						        L.d_in, L.in_image_width, L.in_channels,
						        L.out_channels, L.conv_size,
						        L.d_sum_weight_grad, L.d_sum_bias_grad);
					}
				}
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Backprop Convolve 1: %f ms\n", l, time);
#endif
				if(prev_grad)
				{
					cudaEventRecord(m_start, 0);
					if(L.out_image_width == 10)
					{
#ifdef TIMING
						printf("FAST BACKPROP2 - 10\n");
#endif
						int iter = ceil((REAL)L.in_channels*L.in_image_width*L.in_image_width / NUM_THREADS);
						for(int i=0; i < iter; i++)
						{
							int offset = i*NUM_THREADS;
							BackpropConv2Fast<10><<<m_batch_size, NUM_THREADS>>>(offset, L.d_grad, L.d_out_deriv, L.d_weights,
							        L.d_in, L.in_image_width, L.in_channels,
							        L.out_channels, L.conv_size,
							        prev_grad);
						}
					}
					else if(L.out_image_width == 6)
					{
#ifdef TIMING
						printf("FAST BACKPROP2 - 6\n");
#endif
						int iter = ceil((REAL)L.in_channels*L.in_image_width*L.in_image_width / NUM_THREADS);
						for(int i=0; i < iter; i++)
						{
							int offset = i*NUM_THREADS;
							BackpropConv2Fast<6><<<m_batch_size, NUM_THREADS>>>(offset, L.d_grad, L.d_out_deriv, L.d_weights,
							        L.d_in, L.in_image_width, L.in_channels,
							        L.out_channels, L.conv_size,
							        prev_grad);
						}
					}
					else
					{
#ifdef TIMING
						printf("SLOW BACKPROP2 - %d\n", L.out_image_width);
#endif
						int iter = ceil((REAL)L.in_channels*L.in_image_width*L.in_image_width / NUM_THREADS);
						for(int i=0; i < iter; i++)
						{
							int offset = i*NUM_THREADS;
							BackpropConv2<<<m_batch_size, NUM_THREADS>>>(offset, L.d_grad, L.d_out_deriv, L.d_weights,
							        L.d_in, L.in_image_width, L.in_channels,
							        L.out_channels, L.conv_size,
							        prev_grad);
						}
					}
#ifdef TIMING
					cudaEventRecord(m_stop, 0);
					cudaEventSynchronize(m_stop);
					cudaEventElapsedTime(&time, m_start, m_stop);
					printf("Layer %d, Backprop Convolve 2: %f ms\n", l, time);
#endif
				}
			}
			else if(L.type == MAX_POOL)
			{
				BackpropMaxPool<<<m_batch_size, L.out_image_width>>>(L.d_grad, L.out_image_width, L.out_channels, L.d_mask, prev_grad);
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Backprop Max Pool: %f ms\n", l, time);
#endif
			}
			else if(L.type == AVG_POOL)
			{
				BackpropAvgPool<<<m_batch_size, L.out_image_width>>>(L.d_grad, L.out_image_width, L.out_channels, L.d_mask, prev_grad);
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Backprop Avg Pool: %f ms\n", l, time);
#endif
			}
			else if(L.type == NN_LINEAR || L.type == NN_RELU || L.type == NN_TANH || L.type == NN_ABS_TANH || L.type == NN_SIGMOID)
			{
				int iter = ceil((REAL)L.out_size/NUM_THREADS);
				// Weights
				for(int i=0; i < iter; i++)
				{
					int thread_offset = i*NUM_THREADS;
					BackpropNN<<<m_batch_size, NUM_THREADS>>>(thread_offset,
					        L.d_grad,
					        L.d_in, L.in_size,
					        L.d_out_deriv, L.out_size,
					        L.d_weights, L.weight_rows, L.weight_cols,
					        L.d_sum_weight_grad, L.d_sum_bias_grad);
				}
				if(prev_grad)
				{
					int iter = ceil((REAL)L.weight_cols/NUM_THREADS);
					for(int i=0; i < iter; i++)
					{
						int thread_offset = i*NUM_THREADS;
						BackpropNN2<<<m_batch_size, NUM_THREADS>>>(thread_offset,
						        L.d_grad,
						        L.d_in, L.in_size,
						        L.d_out_deriv, L.out_size,
						        L.d_weights, L.weight_rows, L.weight_cols,
						        prev_grad);
					}
				}
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Backprop NN: %f ms\n", l, time);
#endif
			}
			else if(L.type == NN_SOFTMAX)
			{
				assert(L.in_size < NUM_THREADS);
				assert(L.d_out);
				assert(d_labels);
				assert(prev_grad);
				Y_minus_target<<<m_batch_size, L.in_size>>>(L.d_out, d_labels, L.in_size, prev_grad);
#ifdef TIMING
				cudaEventRecord(m_stop, 0);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&time, m_start, m_stop);
				printf("Layer %d, Backprop Softmax: %f ms\n", l, time);
#endif
			}
			else
			{
				printf("Layer type not implemented yet\n");
				assert(0);
			}
		}
		// Standard backprop
		for(size_t l=0; l < m_layers.size(); l++)
		{
			Layer &L = m_layers[l];
			if(L.weight_size == 0)
			{
				continue;
			}
#ifdef TIMING
			cudaEventRecord(m_start, 0);
			REAL time;
#endif
			int blocks = ceil((REAL)L.weight_size/NUM_THREADS);
			AvgGrads<<<blocks, NUM_THREADS>>>(L.d_sum_weight_grad, L.weight_size, m_batch_size, L.d_delta_weight);
			blocks = ceil((REAL)L.bias_size/NUM_THREADS);
			AvgGrads<<<blocks, NUM_THREADS>>>(L.d_sum_bias_grad, L.bias_size, m_batch_size, L.d_delta_bias);
			// Weights
			blocks = ceil((REAL)L.weight_size/NUM_THREADS);
			UpdateWeights<<<blocks, NUM_THREADS>>>(L.d_momentum_delta_weight, L.d_delta_weight, L.weight_size, learning_rate, momentum_rate, L.d_weights);
			// Bias
			blocks = ceil((REAL)L.bias_size/NUM_THREADS);
			UpdateWeights<<<blocks, NUM_THREADS>>>(L.d_momentum_delta_bias, L.d_delta_bias, L.bias_size, learning_rate, momentum_rate, L.d_biases);
#ifdef TIMING
			cudaEventRecord(m_stop, 0);
			cudaEventSynchronize(m_stop);
			cudaEventElapsedTime(&time, m_start, m_stop);
			printf("Layer %d, Weight update: %f ms\n", l, time);
#endif
		}
	}

	void CudaConvNN::SaveWeights()
	{
		for(size_t i=0; i < m_layers.size(); i++)
		{
			Layer &L = m_layers[i];
			if(L.weight_size == 0)
			{
				continue;
			}
			char file[128];
			vector <REAL> weights;
			// Weights
			{
				sprintf(file, "layer_%02d_weights.raw", (int)i);
				FILE *fp = fopen(file, "wb+");
				assert(fp);
				L.GetWeights(weights);
				fwrite(&weights[0], weights.size()*sizeof(REAL), 1, fp);
				fclose(fp);
			}
			// Momentum weights
			{
				sprintf(file, "layer_%02d_momentum_weights.raw", (int)i);
				FILE *fp = fopen(file, "wb+");
				assert(fp);
				L.GetMomentumWeights(weights);
				fwrite(&weights[0], weights.size()*sizeof(REAL), 1, fp);
				fclose(fp);
			}
			// Bias
			{
				sprintf(file, "layer_%02d_biases.raw", (int)i);
				FILE *fp = fopen(file, "wb+");
				assert(fp);
				L.GetBiases(weights);
				fwrite(&weights[0], weights.size()*sizeof(REAL), 1, fp);
				fclose(fp);
			}
			// Momentum Bias
			{
				sprintf(file, "layer_%02d_momentum_biases.raw", (int)i);
				FILE *fp = fopen(file, "wb+");
				assert(fp);
				L.GetMomentumBiases(weights);
				fwrite(&weights[0], weights.size()*sizeof(REAL), 1, fp);
				fclose(fp);
			}
		}
	}

	bool CudaConvNN::LoadWeights()
	{
		for(size_t i=0; i < m_layers.size(); i++)
		{
			Layer &L = m_layers[i];
			if(L.weight_size == 0)
			{
				continue;
			}
			char file[128];
			// Weights
			{
				sprintf(file, "layer_%02d_weights.raw", (int)i);
				FILE *fp = fopen(file, "rb");
				if(fp == NULL)
				{
					return false;
				}
				vector <REAL> weights(L.weight_size);
				size_t n = fread(&weights[0], 1, weights.size()*sizeof(REAL), fp);
				assert(n == weights.size()*sizeof(REAL));
				L.SetWeights(&weights[0]);
				fclose(fp);
			}
			// Mometum Weights
			{
				sprintf(file, "layer_%02d_momentum_weights.raw", (int)i);
				FILE *fp = fopen(file, "rb");
				if(fp == NULL)
				{
					return false;
				}
				vector <REAL> weights(L.weight_size);
				size_t n = fread(&weights[0], 1, weights.size()*sizeof(REAL), fp);
				assert(n == weights.size()*sizeof(REAL));
				L.SetMomentumWeights(&weights[0]);
				fclose(fp);
			}
			// Bias
			{
				sprintf(file, "layer_%02d_biases.raw", (int)i);
				FILE *fp = fopen(file, "rb");
				if(fp == NULL)
				{
					return false;
				}
				vector <REAL> weights(L.bias_size);
				size_t n = fread(&weights[0], 1, weights.size()*sizeof(REAL), fp);
				assert(n == weights.size()*sizeof(REAL));
				L.SetBiases(&weights[0]);
				fclose(fp);
			}
			// Momentum Bias
			{
				sprintf(file, "layer_%02d_momentum_biases.raw", (int)i);
				FILE *fp = fopen(file, "rb");
				if(fp == NULL)
				{
					return false;
				}
				vector <REAL> weights(L.bias_size);
				size_t n = fread(&weights[0], 1, weights.size()*sizeof(REAL), fp);
				assert(n == weights.size()*sizeof(REAL));
				L.SetMomentumBiases(&weights[0]);
				fclose(fp);
			}
		}
		return true;
	}

	void CudaConvNN::SetImages(const REAL *data, size_t size)
	{
		assert(d_images);
		assert(size);
		CUDA_SAFE_CALL(cudaMemcpy(d_images, data, size, cudaMemcpyHostToDevice));
	}

	void CudaConvNN::SetLabels(const REAL *data, size_t size)
	{
		assert(d_labels);
		assert(size);
		CUDA_SAFE_CALL(cudaMemcpy(d_labels, data, size, cudaMemcpyHostToDevice));
	}

	void CudaConvNN::CheckValues()
	{
		for(size_t i=0; i < m_layers.size(); i++)
		{
			m_layers[i].CheckValues();
		}
	}

} // namespace
