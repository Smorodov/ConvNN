#ifndef __SETTINGS_H__
#define __SETTINGS_H__

// Used by CUDA code
// Change to double when using the gradient checker
typedef double REAL;

// Dataset
const char* const TRAINING_FILE[] =
{
	"Data/cifar-10-batches-bin/data_batch_1.bin",
	"Data/cifar-10-batches-bin/data_batch_2.bin",
	"Data/cifar-10-batches-bin/data_batch_3.bin",
	"Data/cifar-10-batches-bin/data_batch_4.bin",
	"END"
};

const char VALIDATION_FILE[] = "Data/cifar-10-batches-bin/data_batch_5.bin";
const char TEST_FILE[] = "Data/cifar-10-batches-bin/test_batch.bin";

const int IMAGE_SIZE = 32;
const int CATEGORIES = 10;

#endif // __SETTINGS_H__
