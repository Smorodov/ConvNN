#ifndef __CUDA_CONVNN_H__
#define __CUDA_CONVNN_H__

#include <vector>
#include <cuda_runtime.h>
#include "CUDA_ConvNN_Layer.h"

namespace CNN
{

	class CudaConvNN
	{
		public:
			CudaConvNN();
			~CudaConvNN();

			void Init(int image_width, int channels, int categories, int batch_size);
			void AddLayer(LayerType type, int conv_size, int out_channels); // Convolution
			void AddLayer(LayerType type, int out_size); // Standard neural network hidden layer
			void AddLayer(LayerType type); // Max-pool, SoftMax

			void SetImages(const REAL *data, size_t size);
			void SetLabels(const REAL *data, size_t size);

			void FeedForward();
			void BackProp(REAL learning_rate, REAL momentum_rate);
			size_t TotalMemUsed();
			void SaveWeights();
			bool LoadWeights();

			void CheckValues(); // checks for NaN, inf

		public:
			std::vector <Layer> m_layers;
			int m_batch_size;
			REAL *d_images;
			REAL *d_labels;

		private:
			static const int NUM_THREADS = 256;

			int m_image_width;
			int m_image_channels;
			bool m_momentum_init;
			cudaEvent_t m_start, m_stop;
	};

} // namespace

#endif
