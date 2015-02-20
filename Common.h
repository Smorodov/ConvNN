#ifndef __COMMON_H__
#define __COMMON_H__

#include <vector>
//#include <armadillo>
#include "Settings.h"

#include "cmath"
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define fpu_error(x) (isinf(x) || isnan(x))
struct ImageData
{
	int id; // used to group cropped images
	std::vector <REAL> R;
	std::vector <REAL> G;
	std::vector <REAL> B;
	std::vector <REAL> labels;
};

struct Dataset
{
	int image_width;

	std::vector <ImageData> images;
};

//void LoadData(const char *filename, arma::cube &R, arma::cube &G, arma::cube &B, arma::mat &labels);

void LoadData(const char *filename, Dataset &data);
void CalcMean(Dataset &data, std::vector <REAL> &mean_R, std::vector <REAL> &mean_G, std::vector <REAL> &mean_B);
void CentreData(Dataset &data, std::vector <REAL> &mean_R, std::vector <REAL> &mean_G, std::vector <REAL> &mean_B);

#endif
