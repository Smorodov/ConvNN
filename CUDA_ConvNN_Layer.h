#ifndef __CUDA_CONVNN_LAYER_H__
#define __CUDA_CONVNN_LAYER_H__

#include <vector>
#include "Settings.h"

namespace CNN
{
	// Not all implemented yet
	enum LayerType {NN_LINEAR, NN_SIGMOID, NN_TANH, NN_ABS_TANH, NN_RELU, NN_SOFTMAX,
	                CONV_SIGMOID, CONV_TANH, CONV_ABS_TANH, CONV_LINEAR, CONV_RELU,
	                MAX_POOL, AVG_POOL
	               };

	struct Layer
	{
		Layer();

		void CleanUp();

		void InitConv(LayerType t);
		void InitPool(LayerType t);
		void InitNN(LayerType t);
		void InitSoftMax();
		void InitRandWeights(REAL r);

		void SetWeights(const REAL *weights);
		void SetMomentumWeights(const REAL *weights);
		void SetBiases(const REAL *biases);
		void SetMomentumBiases(const REAL *biases);

		void GetWeights(std::vector<REAL> &weights);
		void GetMomentumWeights(std::vector<REAL> &weights);
		void GetBiases(std::vector<REAL> &biases);
		void GetMomentumBiases(std::vector<REAL> &biases);
		void GetOutput(std::vector <REAL> &output);
		void GetSumWeightGrad(std::vector <REAL> &sum_weight_grad);
		void GetSumBiasGrad(std::vector <REAL> &sum_bias_grad);
		void GetGrad(std::vector <REAL> &grad);
		void GetMask(std::vector <REAL> &mask);
		void GetDeriv(std::vector <REAL> &deriv);

		void ZeroOutput();
		void ZeroSumGrad();
		void ZeroMaxPool();

		// Used by gradient checker
		void SetWeightValue(int idx, REAL v);
		REAL GetWeightValue(int idx);

		void SetBiasValue(int idx, REAL v);
		REAL GetBiasValue(int idx);

		// Debugging
		void RunSumGrads();
		void CheckValues();

		LayerType type;
		int num_images;
		int in_image_width;
		int out_image_width;
		int in_channels;
		int out_channels;
		int conv_size;

		// for NN
		int in_size;
		int out_size;

		REAL *d_in; // no memory allocated, just a pointer
		REAL *d_weights;
		REAL *d_biases;
		REAL *d_out;
		REAL *d_out_deriv;
		REAL *d_mask;
		REAL *d_grad;

		REAL *d_sum_weight_grad;
		REAL *d_sum_bias_grad;

		REAL *d_delta_weight;
		REAL *d_delta_bias;

		REAL *d_momentum_delta_weight;
		REAL *d_momentum_delta_bias;

		REAL *d_tmp;

		// Calculated
		int weight_size;
		int bias_size;
		int out_total_size;
		int out_size_per_sample;
		int weight_rows;
		int weight_cols;
		int grad_size;
		int mask_size;
		size_t total_mem_used;
	};

} // namespace

#endif
