#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CUDA_ConvNN.h"
#include "Common.h"

using namespace std;
using namespace CNN;

void RunCUDA_CNN_1(int epochs, REAL learning_rate, int choice); // 32x32 - horizontal flipping
void RunCUDA_CNN_2(int epochs, REAL learning_rate, int choice); // 24x24 - translation + horizontal flipping

int RunValidation(CudaConvNN &NN, const Dataset &validation);
int RunValidation2(CudaConvNN &NN, const Dataset &validation);
void SaveWeights(const std::string &name, const REAL *weights, int in_channels, int out_channels, int conv_size);
void FlipImages(std::vector <ImageData> &images, int image_size);
void OutputResultsForSite(CudaConvNN &NN, const Dataset &data, vector <REAL> &mean_R, vector <REAL> &mean_G, vector <REAL> &mean_B);
void GradientChecker(CudaConvNN &NN, Dataset &training);
void CropImages(const Dataset &in, Dataset &out);

int main()
{
	int choice, epochs;
	REAL learning_rate = 0.01;
	cout << "What would you like to do?" << endl;
	cout << "1 - train data" << endl;
	cout << "2 - run network on test set" << endl;
	cout << "or any other value to exit" << endl;
	cout << endl;
	cout << "choice: ";
	cin >> choice;
	if(choice < 1 || choice > 2)
	{
		return 1;
	}
	if(choice == 1)
	{
		cout << "How many epochs to train for: ";
		cin >> epochs;
		if(epochs <= 0)
		{
			cout << "Invalid number of epochs" << endl;
			return 1;
		}
		int choice2;
		cout << "What learning rate?" << endl;
		cout << "1 - 0.01" << endl;
		cout << "2 - 0.001" << endl;
		cout << "3 - 0.0001" << endl;
		cout << endl;
		cout << "choice: ";
		cin >> choice2;
		if(choice2 < 1 || choice2 > 3)
		{
			cout << "Invalid choice" << endl;
			return 1;
		}
		if(choice2 == 1)
		{
			learning_rate = 0.01;
		}
		else if(choice2 == 2)
		{
			learning_rate = 0.001;
		}
		else
		{
			learning_rate = 0.0001;
		}
	}
	//RunCUDA_CNN_1(epochs, learning_rate, choice); // horizontal flipping only
	RunCUDA_CNN_2(epochs, learning_rate, choice); // translation + horizontal flipping
	return 0;
}

void RunCUDA_CNN_1(int epochs, REAL learning_rate, int choice)
{
	Dataset validation, testing;
	vector <REAL> mean_R, mean_G, mean_B;
	int image_width, categories;
	// Load and pre-process data
	{
		LoadData(VALIDATION_FILE, validation);
		image_width = validation.image_width;
		categories = validation.images[0].labels.size();
		assert(image_width == 32);
		assert(categories == 10);
		// Mean subtract
		CalcMean(validation, mean_R, mean_G, mean_B);
		CentreData(validation, mean_R, mean_G, mean_B);
	}
	// Training parameters
	const int batch_size = 64;
	const REAL momentum_rate = 0.9;
	vector <REAL> batch(batch_size * image_width * image_width * 3);
	vector <REAL> batch_labels(batch_size * categories);
	vector <REAL> ret_weights;
	// Construction of the network architecture
	CudaConvNN NN;
	NN.Init(image_width, 3, categories, batch_size);
	//32x32 input
	NN.AddLayer(CONV_RELU, 5, 32);
	NN.AddLayer(AVG_POOL);
	NN.AddLayer(CONV_RELU, 5, 32);
	NN.AddLayer(AVG_POOL);
	NN.AddLayer(CONV_RELU, 4, 64);
	NN.AddLayer(AVG_POOL);
	NN.AddLayer(NN_RELU, 64);
	NN.AddLayer(NN_LINEAR, categories);
	NN.AddLayer(NN_SOFTMAX);
	// Initialise with Gaussian weights using C++11 stuff
	// This code is moved here because the CUDA compiler doesn't handle C++11
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		gen.seed(time(NULL)); // random seed
		std::normal_distribution<> dice(0, 0.1); // 0.01 doesn't work at all!!! Something very picky about the weights ...
		for(size_t i=0; i < NN.m_layers.size(); i++)
		{
			Layer &L = NN.m_layers[i];
			if(L.weight_size == 0)
			{
				continue;
			}
			std::vector <REAL> weights(L.weight_size);
			for(size_t j=0; j < weights.size(); j++)
			{
				weights[j] = dice(gen);
			}
			L.SetWeights(&weights[0]);
		}
	}
	//GradientChecker(NN, validation);
	FILE *fp = NULL; // for graphing
	if(choice == 1)
	{
		char ch;
		cout << "Do you want to try and load existing saved weights? y/n: ";
		cin >> ch;
		if(tolower(ch) == 'y')
		{
			fp = fopen("error_plot.txt", "a+");
			assert(fp);
			if(NN.LoadWeights())
			{
				cout << "Found existing weights" << endl;
			}
			else
			{
				cout << "No existing weights found" << endl;
			}
		}
		else
		{
			fp = fopen("error_plot.txt", "w+");
		}
	}
	else if(choice == 2)
	{
		if(NN.LoadWeights() == false)
		{
			cout << "No existing weights found" << endl;
			return;
		}
	}
	// Testing error
	if(choice == 2)
	{
		LoadData(TEST_FILE, testing);
		CropImages(testing, testing);
		CentreData(testing, mean_R, mean_G, mean_B);
		chrono::time_point<chrono::system_clock> start, end;
		start = chrono::system_clock::now();
		int err = RunValidation(NN, testing);
		end = chrono::system_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		cout << "Testing error: " << err << " time=" << elapsed << " ms" << endl;
		//OutputResultsForSite(NN, testing, mean_R, mean_G, mean_B);
		return;
	}
	printf("Total MB used: %.2f\n", NN.TotalMemUsed() / 1000000.0);
	for(int e=0; e < epochs; e++)
	{
		printf("epoch %d/%d\n", e+1, epochs);
		// Go through each training file
		// Save memory loading individually rather than all in one shot
		for(int d=0; string(TRAINING_FILE[d]) != "END"; d++)
		{
			Dataset training;
			LoadData(TRAINING_FILE[d], training);
			CentreData(training, mean_R, mean_G, mean_B);
			// Create more training images by horizontal flipping
			vector <ImageData> images2 = training.images;
			FlipImages(images2, training.image_width);
			training.images.insert(training.images.end(), images2.begin(), images2.end());
			random_shuffle(training.images.begin(), training.images.end());
			int iterations = ceil((REAL)training.images.size() / batch_size);
			for(int iter=0; iter < iterations; iter++)
			{
				int offset = iter*batch_size;
				int total_images = 0;
				for(int b=0; b < batch_size; b++)
				{
					int idx = offset + b;
					if(idx >= (int)training.images.size())
					{
						break;
					}
					total_images = b;
					int bidx = b * image_width * image_width * 3;
					REAL *Rdst = &batch[bidx];
					REAL *Gdst = &batch[bidx + image_width*image_width];
					REAL *Bdst = &batch[bidx + image_width*image_width*2];
					REAL *Ldst = &batch_labels[b * categories];
					memcpy(Rdst, &training.images[idx].R[0], image_width * image_width * sizeof(REAL));
					memcpy(Gdst, &training.images[idx].G[0], image_width * image_width * sizeof(REAL));
					memcpy(Bdst, &training.images[idx].B[0], image_width * image_width * sizeof(REAL));
					memcpy(Ldst, &training.images[idx].labels[0], categories * sizeof(REAL));
				}
				NN.SetImages(&batch[0], batch.size()*sizeof(REAL));
				NN.SetLabels(&batch_labels[0], batch_labels.size()*sizeof(REAL));
				total_images++; // off by 1
				NN.m_batch_size = total_images;
				NN.FeedForward();
				NN.BackProp(learning_rate, momentum_rate);
				printf("    training set=%d %d/%d\n", d, iter+1, iterations);
			}
			printf("Saving ...\n");
			NN.SaveWeights();
			if(NN.m_layers[0].type == CONV_RELU)
			{
				NN.m_layers[0].GetWeights(ret_weights);
				SaveWeights("weights.png", &ret_weights[0],  NN.m_layers[0].in_channels,  NN.m_layers[0].out_channels,  NN.m_layers[0].conv_size);
			}
			chrono::time_point<chrono::system_clock> start, end;
			start = chrono::system_clock::now();
			int err = RunValidation(NN, validation);
			end = chrono::system_clock::now();
			int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
			cout << "Validation error: " << err << " time=" << elapsed << " ms" << endl;
			fprintf(fp, "%d\n", err);
			fflush(fp);
		}
	}
	fclose(fp);
}

void RunCUDA_CNN_2(int epochs, REAL learning_rate, int choice)
{
	// This version uses translation + horizontal flipping to generate more training images
	Dataset validation, testing;
	vector <REAL> mean_R, mean_G, mean_B;
	int image_width, categories;
	// Load and pre-process data
	{
		LoadData(VALIDATION_FILE, validation);
		CropImages(validation, validation);
		// Create more images by horizontal flipping
		vector <ImageData> images2 = validation.images;
		FlipImages(images2, validation.image_width);
		validation.images.insert(validation.images.end(), images2.begin(), images2.end());
		image_width = validation.image_width;
		categories = validation.images[0].labels.size();
		assert(image_width == 24);
		assert(categories == 10);
		// Mean subtract
		CalcMean(validation, mean_R, mean_G, mean_B);
		CentreData(validation, mean_R, mean_G, mean_B);
	}
	// Training parameters
	const int batch_size = 64;																		// << - BATCH SIZE
	const REAL momentum_rate = 0.9;
	vector <REAL> batch(batch_size * image_width * image_width * 3);
	vector <REAL> batch_labels(batch_size * categories);
	vector <REAL> ret_weights;
	// Construction of the network architecture
	CudaConvNN NN;
	NN.Init(24, 3, categories, batch_size);
	// 24x24 input
	NN.AddLayer(CONV_RELU, 5, 64);
	NN.AddLayer(MAX_POOL);
	NN.AddLayer(CONV_RELU, 5, 64);
	NN.AddLayer(MAX_POOL);
	NN.AddLayer(CONV_RELU, 3, 64);
	NN.AddLayer(NN_RELU, 64);
	NN.AddLayer(NN_LINEAR, categories);
	NN.AddLayer(NN_SOFTMAX);
	// Initialise with Gaussian weights using C++11 stuff
	// This has to be done here becuse CUDA compiler doesn't handle C++11
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		gen.seed(time(NULL));
		// 0.01 doesn't work at all!!! Not enough precision with REAL?
		// There is something very sensitive about the weights set
		std::normal_distribution<> dice(0, 0.1);
		for(size_t i=0; i < NN.m_layers.size(); i++)
		{
			Layer &L = NN.m_layers[i];
			if(L.weight_size == 0)
			{
				continue;
			}
			std::vector <REAL> weights(L.weight_size);
			for(size_t j=0; j < weights.size(); j++)
			{
				weights[j] = dice(gen);
			}
			L.SetWeights(&weights[0]);
		}
	}
	//  GradientChecker(NN, validation);
	FILE *fp = NULL; // for graphing
	if(choice == 1)
	{
		char ch;
		cout << "Do you want to try and load existing saved weights? y/n: ";
		cin >> ch;
		if(tolower(ch) == 'y')
		{
			fp = fopen("error_plot.txt", "a+");
			assert(fp);
			if(NN.LoadWeights())
			{
				cout << "Found existing weights" << endl;
			}
			else
			{
				cout << "No existing weights found" << endl;
			}
		}
		else
		{
			fp = fopen("error_plot.txt", "w+");
		}
	}
	else if(choice == 2)
	{
		if(NN.LoadWeights() == false)
		{
			cout << "No existing weights found" << endl;
			return;
		}
	}
	// Testing error
	if(choice == 2)
	{
		LoadData(TEST_FILE, testing);
		CropImages(testing, testing);
		CentreData(testing, mean_R, mean_G, mean_B);
		chrono::time_point<chrono::system_clock> start, end;
		start = chrono::system_clock::now();
		int err = RunValidation2(NN, testing);
		end = chrono::system_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		cout << "Testing error: " << err << " time=" << elapsed << " ms" << endl;
		//OutputResultsForSite(NN, testing, mean_R, mean_G, mean_B);
		return;
	}
	printf("Total MB used: %.2f\n", NN.TotalMemUsed() / 1000000.0);
	for(int e=0; e < epochs; e++)
	{
		printf("epoch %d/%d\n", e+1, epochs);
		// Go through each training file
		// Save memory loading individually rather than all in one shot
		for(int d=0; string(TRAINING_FILE[d]) != "END"; d++)
		{
			Dataset training;
			LoadData(TRAINING_FILE[d], training);
			CropImages(training, training);
			CentreData(training, mean_R, mean_G, mean_B);
			// Create more training images by horizontal flipping
			vector <ImageData> images2 = training.images;
			FlipImages(images2, training.image_width);
			training.images.insert(training.images.end(), images2.begin(), images2.end());
			random_shuffle(training.images.begin(), training.images.end());
			int iterations = ceil((REAL)training.images.size() / (REAL)batch_size);
			//iterations /= 10;
			for(int iter=0; iter < iterations; iter++)
			{
				int offset = iter*batch_size;
				int total_images = 0;
				for(int b=0; b < batch_size; b++)
				{
					int idx = offset + b;
					if(idx >= (int)training.images.size())
					{
						break;
					}
					total_images = b;
					int bidx = b * image_width * image_width * 3;
					REAL *Rdst = &batch[bidx];
					REAL *Gdst = &batch[bidx + image_width*image_width];
					REAL *Bdst = &batch[bidx + image_width*image_width*2];
					REAL *Ldst = &batch_labels[b * categories];
					memcpy(Rdst, &training.images[idx].R[0], image_width * image_width * sizeof(REAL));
					memcpy(Gdst, &training.images[idx].G[0], image_width * image_width * sizeof(REAL));
					memcpy(Bdst, &training.images[idx].B[0], image_width * image_width * sizeof(REAL));
					memcpy(Ldst, &training.images[idx].labels[0], categories * sizeof(REAL));
				}
				NN.SetImages(&batch[0], batch.size()*sizeof(REAL));
				NN.SetLabels(&batch_labels[0], batch_labels.size()*sizeof(REAL));
				total_images++; // off by 1
				NN.m_batch_size = total_images;
				NN.FeedForward();
				NN.BackProp(learning_rate, momentum_rate);
				printf("    training set=%d %d/%d\n", d, iter+1, iterations);
			}
			printf("Saving ...\n");
			NN.SaveWeights();
			if(NN.m_layers[0].type == CONV_RELU)
			{
				NN.m_layers[0].GetWeights(ret_weights);
				SaveWeights("weights.png", &ret_weights[0],  NN.m_layers[0].in_channels,  NN.m_layers[0].out_channels,  NN.m_layers[0].conv_size);
			}
			chrono::time_point<chrono::system_clock> start, end;
			printf("Validation ...\n");
			start = chrono::system_clock::now();
			int err = RunValidation2(NN, validation);
			end = chrono::system_clock::now();
			int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
			cout << "Validation error: " << err << " time=" << elapsed << " ms" << endl;
			fprintf(fp, "%d\n", err);
			fflush(fp);
		}
	}
	fclose(fp);
}

void GradientChecker(CudaConvNN &NN, Dataset &training)
{
	// Gradient checker
	int image_width = training.image_width;
	int categories = training.images[0].labels.size();
	vector <REAL> batch(image_width * image_width * 3);
	vector <REAL> batch_labels(categories);
	NN.m_batch_size = 1;
	REAL step = 0.001;
	REAL learning_rate = 0.01;
	REAL momentum_rate = 0;
	if(sizeof(REAL) == sizeof(double))
	{
		step = 0.00001;
	}
	else
	{
		cout << "You're not using double type, it is recommended that you do so for gradient checker" << endl;
		cout << "Edit Settings.h if your CUDA card supports double" << endl;
	}
	// Select the gradient of interest
	int layer_idx = 0;
	int grad_idx = 1;
	cout << "Running gradient checker ..." << endl;
	for(size_t idx=0; idx < training.images.size(); idx++)
	{
		REAL *Rdst = &batch[0];
		REAL *Gdst = &batch[image_width*image_width];
		REAL *Bdst = &batch[image_width*image_width*2];
		REAL *Ldst = &batch_labels[0];
		memcpy(Rdst, &training.images[idx].R[0], image_width * image_width * sizeof(REAL));
		memcpy(Gdst, &training.images[idx].G[0], image_width * image_width * sizeof(REAL));
		memcpy(Bdst, &training.images[idx].B[0], image_width * image_width * sizeof(REAL));
		memcpy(Ldst, &training.images[idx].labels[0], categories * sizeof(REAL));
		NN.SetImages(&batch[0], image_width * image_width * 3 * sizeof(REAL));
		NN.SetLabels(&batch_labels[0], categories * sizeof(REAL));
		vector <REAL> output;
		REAL orig = NN.m_layers[layer_idx].GetWeightValue(grad_idx);
		REAL neg = orig - step;
		REAL plus = orig + step;
		////////////////////////////////////////////////
		NN.m_layers[layer_idx].SetWeightValue(grad_idx, neg);
		NN.FeedForward();
		NN.m_layers.back().GetOutput(output);
		double E1 = 0;
		for(int i=0; i < categories; i++)
		{
			E1 += batch_labels[i] * log(output[i]);
		}
		E1 = -E1;
		////////////////////////////////////////////////
		NN.m_layers[layer_idx].SetWeightValue(grad_idx, plus);
		NN.FeedForward();
		NN.m_layers.back().GetOutput(output);
		double E2 = 0;
		for(int i=0; i < categories; i++)
		{
			E2 += batch_labels[i] * log(output[i]);
		}
		E2 = -E2;
		////////////////////////////////////////////////
		double finite_gradient = (E2 - E1)/(2.0*step);
		NN.m_layers[layer_idx].SetWeightValue(grad_idx, orig);
		NN.FeedForward();
		NN.BackProp(learning_rate, momentum_rate);
		vector <REAL> grads;
		NN.m_layers[layer_idx].GetSumWeightGrad(grads);
		double diff = fabs(finite_gradient - grads[grad_idx]);
		double err = diff / min((REAL)fabs(finite_gradient), fabs(grads[grad_idx]));
		if(err > 0.001)
		{
			cerr << "LARGE ERROR: at idx=" << idx << " finite_gradient=" << finite_gradient << " --- calc=" << grads[grad_idx] << " ---- " << "err=" << err << endl;
		}
	}
}

int RunValidation(CudaConvNN &NN, const Dataset &validation)
{
	// Validation
	int categories = validation.images[0].labels.size();
	int image_width = validation.image_width;
	int batch_size = NN.m_batch_size;
	assert(categories == 10);
	int iter = ceil((REAL)validation.images.size() / batch_size);
	vector <REAL> batch(batch_size * image_width * image_width * 3);
	vector <REAL> batch_labels(batch_size * categories);
	vector <REAL> softmax_out;
	int correct = 0;
	int error = 0;
	for(int i=0; i < iter; i++)
	{
		for(int b=0; b < batch_size; b++)
		{
			int idx = i*batch_size + b;
			if(idx >= (int)validation.images.size())
			{
				break;
			}
			int bidx = b * image_width * image_width * 3;
			REAL *Rdst = &batch[bidx];
			REAL *Gdst = &batch[bidx + image_width*image_width];
			REAL *Bdst = &batch[bidx + image_width*image_width*2];
			REAL *Ldst = &batch_labels[b * categories];
			memcpy(Rdst, &validation.images[idx].R[0], image_width * image_width * sizeof(REAL));
			memcpy(Gdst, &validation.images[idx].G[0], image_width * image_width * sizeof(REAL));
			memcpy(Bdst, &validation.images[idx].B[0], image_width * image_width * sizeof(REAL));
			memcpy(Ldst, &validation.images[idx].labels[0], categories * sizeof(REAL));
		}
		NN.SetImages(&batch[0], batch.size()*sizeof(REAL));
		NN.SetLabels(&batch_labels[0], batch_labels.size()*sizeof(REAL));
		NN.FeedForward();
		NN.m_layers.back().GetOutput(softmax_out);
		for(int b=0; b < batch_size; b++)
		{
			int idx = i*batch_size + b;
			if(idx >= (int)validation.images.size())
			{
				break;
			}
			const REAL *l = &validation.images[idx].labels[0];
			REAL *softmax = &softmax_out[b * categories];
			REAL max_val = -FLT_MAX;
			int max_at = 0;
			int label_pos = -1;
			for(int c=0; c < categories; c++)
			{
				assert(isnan(softmax[c]) == false);
				//assert(softmax[c] >= 0);
				if(softmax[c] > max_val)
				{
					max_val = softmax[c];
					max_at = c;
				}
				if(l[c] == 1)
				{
					label_pos = c;
				}
			}
			cout << "label_pos = " << label_pos << endl;
			assert(label_pos != -1);
			if(max_at == label_pos)
			{
				correct++;
			}
			else
			{
				error++;
			}
		}
	}
	return error;
}

int RunValidation2(CudaConvNN &NN, const Dataset &validation)
{
	// Validation
	int categories = validation.images[0].labels.size();
	int image_width = validation.image_width;
	int batch_size = NN.m_batch_size;
	assert(image_width == 24);
	assert(categories == 10);
	int iter = ceil((REAL)validation.images.size() / batch_size);
	vector <REAL> batch(batch_size * image_width * image_width * 3);
	vector <REAL> batch_labels(batch_size * categories);
	vector <REAL> softmax_out;
	// Find out the original number of images before cropping
	int max_id = 0;
	for(size_t i=0; i < validation.images.size(); i++)
	{
		max_id = max(max_id, validation.images[i].id);
	}
	max_id++;
	vector <vector<REAL>> tally_softmax(max_id);
	vector <vector<REAL>> orig_labels(max_id);
	for(size_t i=0; i < tally_softmax.size(); i++)
	{
		tally_softmax[i].resize(categories, 0);
	}
	for(int i=0; i < iter; i++)
	{
		for(int b=0; b < batch_size; b++)
		{
			int idx = i*batch_size + b;
			if(idx >= (int)validation.images.size())
			{
				break;
			}
			int orig_image = validation.images[idx].id;
			orig_labels[orig_image] = validation.images[idx].labels;
			int bidx = b * image_width * image_width * 3;
			REAL *Rdst = &batch[bidx];
			REAL *Gdst = &batch[bidx + image_width*image_width];
			REAL *Bdst = &batch[bidx + image_width*image_width*2];
			REAL *Ldst = &batch_labels[b * categories];
			memcpy(Rdst, &validation.images[idx].R[0], image_width * image_width * sizeof(REAL));
			memcpy(Gdst, &validation.images[idx].G[0], image_width * image_width * sizeof(REAL));
			memcpy(Bdst, &validation.images[idx].B[0], image_width * image_width * sizeof(REAL));
			memcpy(Ldst, &validation.images[idx].labels[0], categories * sizeof(REAL));
		}
		NN.SetImages(&batch[0], batch.size()*sizeof(REAL));
		NN.SetLabels(&batch_labels[0], batch_labels.size()*sizeof(REAL));
		NN.FeedForward();
		NN.m_layers.back().GetOutput(softmax_out);
		for(int b=0; b < batch_size; b++)
		{
			int idx = i*batch_size + b;
			if(idx >= (int)validation.images.size())
			{
				break;
			}
			int orig_image = validation.images[idx].id;
			REAL *softmax = &softmax_out[b * categories];
			for(int c=0; c < categories; c++)
			{
				assert(isnan(softmax[c]) == false);
				tally_softmax[orig_image][c] += softmax[c];
			}
		}
	}
	int correct = 0;
	int error = 0;
	for(size_t i=0; i < tally_softmax.size(); i++)
	{
		vector <REAL> &softmax = tally_softmax[i];
		vector <REAL> &labels = orig_labels[i];
		REAL max_val = -FLT_MAX;
		int max_at = 0;
		int label_pos = -1;
		assert(labels.size());
		assert(softmax.size());
		//	cout << "labels = ";
		for(size_t c=0; c < softmax.size(); c++)
		{
			if(softmax[c] > max_val)
			{
				max_val = softmax[c];
				max_at = c;
			}
			//		cout << labels[c] << ",";
			if(labels[c] == 1)
			{
				label_pos = c;
			}
		}
		//  cout  <<  endl;
		//	cout << "label_pos = " << label_pos << endl;
		assert(label_pos != -1);
		if(max_at == label_pos)
		{
			correct++;
		}
		else
		{
			error++;
		}
	}
	return error;
}

void SaveWeights(const std::string &name, const REAL *weights, int in_channels, int out_channels, int conv_size)
{
	int images_per_row = ceil(sqrt(out_channels));
	int scale = 5;
	int gap = 2;
	int canvas_width = (conv_size+gap)*images_per_row;
	int canvas_height = canvas_width;
	cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
	REAL low = FLT_MAX;
	REAL high = -FLT_MAX;
	for(int i=0; i < in_channels*out_channels*conv_size*conv_size; i++)
	{
		low = min(low, weights[i]);
		high = max(high, weights[i]);
	}
	for(int j=0; j < out_channels; j++)
	{
		int y = j/images_per_row;
		int x = j - y*images_per_row;
		y *= (conv_size + gap);
		x *= (conv_size + gap);
		for(int i=0; i < in_channels; i++)
		{
			int idx = i*out_channels + j;
			const REAL *w = &weights[idx * conv_size * conv_size];
			for(int a=0; a < conv_size; a++)
			{
				for(int b=0; b < conv_size; b++)
				{
					assert(!isnan(w[a*conv_size+b]));
					REAL v = 255*(w[a*conv_size+b] - low) / (high - low);
					if(v < 0)
					{
						v = 0;
					}
					if(v > 255)
					{
						v = 255;
					}
					canvas.at<cv::Vec3b>(y+a, x+b)[2-i] = v; // BGR not RGB
				}
			}
		}
	}
	cv::resize(canvas, canvas, cv::Size(canvas_width*scale, canvas_width*scale), 0, 0, cv::INTER_NEAREST);
	cv::imwrite(name, canvas);
}

void FlipImages(std::vector <ImageData> &images, int image_size)
{
	for(size_t i=0; i < images.size(); i++)
	{
		for(int y=0; y < image_size; y++)
		{
			for(int x=0; x < image_size/2; x++)
			{
				swap(images[i].R[y*image_size+x], images[i].R[y*image_size + (image_size-x-1)]);
				swap(images[i].G[y*image_size+x], images[i].G[y*image_size + (image_size-x-1)]);
				swap(images[i].B[y*image_size+x], images[i].B[y*image_size + (image_size-x-1)]);
			}
		}
	}
}

void OutputResultsForSite(CudaConvNN &NN, const Dataset &data, vector <REAL> &mean_R, vector <REAL> &mean_G, vector <REAL> &mean_B)
{
	const char *labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
	const int num_results = 30; // for correct and incorrect results
	int categories = data.images[0].labels.size();
	int image_width = data.image_width;
	int batch_size = 64;
	int iter = ceil((REAL)data.images.size() / batch_size);
	vector <REAL> batch(batch_size * image_width * image_width * 3);
	vector <REAL> batch_labels(batch_size * categories);
	vector <REAL> softmax_out;
	int correct = 0;
	int error = 0;
	for(int i=0; i < iter; i++)
	{
		for(int b=0; b < batch_size; b++)
		{
			int idx = i*batch_size + b;
			if(idx >= (int)data.images.size())
			{
				break;
			}
			int bidx = b * image_width * image_width * 3;
			REAL *Rdst = &batch[bidx];
			REAL *Gdst = &batch[bidx + image_width*image_width];
			REAL *Bdst = &batch[bidx + image_width*image_width*2];
			REAL *Ldst = &batch_labels[b * categories];
			memcpy(Rdst, &data.images[idx].R[0], image_width * image_width * sizeof(REAL));
			memcpy(Gdst, &data.images[idx].G[0], image_width * image_width * sizeof(REAL));
			memcpy(Bdst, &data.images[idx].B[0], image_width * image_width * sizeof(REAL));
			memcpy(Ldst, &data.images[idx].labels[0], categories * sizeof(REAL));
		}
		NN.SetImages(&batch[0], batch.size()*sizeof(REAL));
		NN.SetLabels(&batch_labels[0], batch_labels.size()*sizeof(REAL));
		NN.FeedForward();
		NN.m_layers.back().GetOutput(softmax_out);
		for(int b=0; b < batch_size; b++)
		{
			int idx = i*batch_size + b;
			if(idx >= (int)data.images.size())
			{
				break;
			}
			const REAL *l = &data.images[idx].labels[0];
			REAL *softmax = &softmax_out[b * categories];
			REAL max_val = -FLT_MAX;
			int max_at = 0;
			int label_pos = -1;
			for(int c=0; c < categories; c++)
			{
				assert(!isnan(softmax[c]));
				//  assert(softmax[c] >= 0);
				if(softmax[c] > max_val)
				{
					max_val = softmax[c];
					max_at = c;
				}
				if(l[c] == 1)
				{
					label_pos = c;
				}
			}
			assert(label_pos != -1);
			char file[256];
			if(max_at == label_pos)
			{
				sprintf(file, "correct-%02d.png", correct);
				correct++;
				if(correct >= num_results)
				{
					continue;
				}
			}
			else
			{
				sprintf(file, "error-%02d.png", error);
				error++;
				if(error >= num_results)
				{
					continue;
				}
			}
			// Draw
			cv::Mat img(image_width, image_width, CV_8UC3);
			int k=0;
			for(int y=0; y < image_width; y++)
			{
				for(int x=0; x < image_width; x++)
				{
					img.at<cv::Vec3b>(y,x)[0] = (data.images[idx].B[k] + mean_B[k])*255;
					img.at<cv::Vec3b>(y,x)[1] = (data.images[idx].G[k] + mean_G[k])*255;
					img.at<cv::Vec3b>(y,x)[2] = (data.images[idx].R[k] + mean_R[k])*255;
					k++;
				}
			}
			cv::resize(img, img, cv::Size(128,128), 0, 0, cv::INTER_NEAREST);
			cv::Mat canvas = cv::Mat::zeros(325, 140, CV_8UC3);
			img.copyTo(canvas(cv::Rect(0,0,128,128)));
			for(int c=0; c < categories; c++)
			{
				int yy = c*20 + 142;
				char text[256];
				sprintf(text, "%s: %.2f", labels[c], softmax[c]);
				if(max_at == c)
				{
					cv::putText(canvas, text, cv::Point(1, yy), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0));
				}
				else
				{
					cv::putText(canvas, text, cv::Point(1, yy), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255));
				}
			}
			cv::imwrite(file, canvas);
			if(error >= num_results+1 && correct >= num_results+1)
			{
				return;
			}
		}
	}
}

void CropImages(const Dataset &in, Dataset &out)
{
	const int out_width = 24;
	int in_width = in.image_width;
	Dataset ret;
	ret.image_width = out_width;
	ret.images.resize(in.images.size()*9);
	int idx=0;
	for(size_t i=0; i < in.images.size(); i++)
	{
		for(int j=0; j < 3; j++)
		{
			for(int k=0; k < 3; k++)
			{
				ret.images[idx].id = i;
				ret.images[idx].R.resize(out_width*out_width);
				ret.images[idx].G.resize(out_width*out_width);
				ret.images[idx].B.resize(out_width*out_width);
				ret.images[idx].labels = in.images[i].labels;
				int ystart = j*4;
				int xstart = k*4;
				for(int y=0; y < out_width; y++)
				{
					for(int x=0; x < out_width; x++)
					{
						ret.images[idx].R[y*out_width + x] = in.images[i].R[(ystart + y)*in_width + xstart + x];
						ret.images[idx].G[y*out_width + x] = in.images[i].G[(ystart + y)*in_width + xstart + x];
						ret.images[idx].B[y*out_width + x] = in.images[i].B[(ystart + y)*in_width + xstart + x];
					}
				}
				idx++;
			}
		}
	}
	out = ret;
}
