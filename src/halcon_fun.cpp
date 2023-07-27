#include "halcon_fun.h"

#include "mycv.hpp"

namespace mycv {


#ifdef HAS_HALCON




int  CreateHalconShapeModel(HalconCpp::HImage& temp, double start_angle, double end_angle, HalconCpp::HTuple& model_id)
{
	//printf("in \n");
	if (!temp.IsInitialized())
	{
		printf("img is empty!\n");
		return -1;
	}
	//angleRange 模型的搜索角度范围
	double start = start_angle * PI / 180;//角度制转弧度制
	double extend = (end_angle - start_angle) * PI / 180;//搜索范围
	HalconCpp::HTuple	NumLevels = 3;
	HalconCpp::HTuple	AngleStart(start);
	HalconCpp::HTuple	AngleExtent(extend);
	HalconCpp::HTuple	AngleStep = "auto";
	HalconCpp::HTuple	Optimization = "auto";
	HalconCpp::HTuple	Metric = "use_polarity";
	HalconCpp::HTuple	Contrast = "auto";
	HalconCpp::HTuple	MinContrast = "auto";

	HalconCpp::CreateShapeModel(temp, NumLevels, AngleStart, AngleExtent, AngleStep,
		Optimization, Metric, Contrast, MinContrast, &model_id);

	return error_code::kSuccess;
}

int FindHalconShapeModel(HalconCpp::HImage& img, const HalconCpp::HTuple& model_id, double start_angle, double end_angle, double result[4])
{
	double start = start_angle * PI / 180;//角度制转弧度制
	double extend = (end_angle - start_angle) * PI / 180;//搜索范围
	HalconCpp::HTuple	NumLevels = "auto";
	HalconCpp::HTuple	AngleStart(start);
	HalconCpp::HTuple	AngleExtent(extend);
	HalconCpp::HTuple MinScore = 0.5;
	HalconCpp::HTuple SubPixel = "interpolation";
	HalconCpp::HTuple row, col, score, angle;

	//center x
	result[0] = 0;
	//center y
	result[1] = 0;
	//finalAngle
	result[2] = 0;
	//finalVal(percent)
	result[3] = 0;


	HalconCpp::FindShapeModel(img, model_id, AngleStart, AngleExtent, MinScore, 1, 0.5, SubPixel, NumLevels, 0.9,
		&row, &col, &angle, &score);
	//center x
	result[0] = col.D();
	//center y
	result[1] = row.D();
	//finalAngle
	result[2] = angle.D();
	//finalVal(percent)
	result[3] = score.D();


	return error_code::kSuccess;
}


int CvMat2HalconImage(const cv::Mat& mat_image, HalconCpp::HImage& halcon_image)
{
	CHECK_RET(CheckImageEmpty(mat_image));
	int width = mat_image.cols;
	int height = mat_image.rows;
	
	if (1 == mat_image.channels())
	{
		if (mat_image.isContinuous())
			HalconCpp::GenImage1(&halcon_image, "byte", width, height, (uint8_t)mat_image.data);
	}

	return error_code::kSuccess;
}


void createHalconImage(BmpData& bmpData, HalconCpp::HImage& img, int channel)
{
	//stride 是图像矩阵的宽度，例如图片宽100pixels，有4通道，则stride=100*4；
	int width = bmpData.width, height = bmpData.height, stride = bmpData.stride;
	byte* imdata = bmpData.data;
	int size = width * height;
	byte* R;
	byte* G;
	byte* B;
	byte* gray;
	switch (channel)
	{
		//red channel
	case 0:
		R = new byte[size];
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				R[i * width + j] = imdata[i * width * 4 + 4 * j + 2];
			}
		}
		HalconCpp::GenImage1(&img, "byte", width, height, (Hlong)R);
		delete[] R;
		break;

		//green channel
	case 1:
		G = new byte[size];
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				G[i * width + j] = imdata[i * width * 4 + 4 * j + 1];
			}
		}
		HalconCpp::GenImage1(&img, "byte", width, height, (Hlong)G);
		delete[] G;
		break;

		//blue channel
	case 2:
		B = new byte[size];
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				B[i * width + j] = imdata[i * width * 4 + 4 * j];
			}
		}
		HalconCpp::GenImage1(&img, "byte", width, height, (Hlong)B);
		delete[] B;
		break;

		//gray channel
	case 3:
		/*R = new byte[size];
		G = new byte[size];
		B = new byte[size];*/
		gray = new byte[size];
		//灰度图转化心理学公式 Grey = 0.299*R + 0.587*G + 0.114*B
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				/*B[i*width + j] = imdata[i*width * 4 + 4 * j];
				G[i*width + j] = imdata[i*width * 4 + 4 * j + 1];
				R[i*width + j] = imdata[i*width * 4 + 4 * j + 2];*/
				gray[i * width + j] = byte(0.299 * imdata[i * width * 4 + 4 * j] + 0.587 * imdata[i * width * 4 + 4 * j + 1] + 0.114 * imdata[i * width * 4 + 4 * j + 2]);
			}
		}
		//HalconCpp::GenImage3(&img, "byte", width, height, (Hlong)R, (Hlong)G, (Hlong)B);
		HalconCpp::GenImage1(&img, "byte", width, height, (Hlong)gray);
		//HalconCpp::WriteImage(img, "png", 255, "G:/geay.png");
		/*delete[] R;
		delete[] G;
		delete[] B;*/
		delete[] gray;
		break;

	case 4:
		//rgb color image
		R = new byte[size];
		G = new byte[size];
		B = new byte[size];
		//灰度图转化公式 Grey = 0.299*R + 0.587*G + 0.114*B
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				B[i * width + j] = imdata[i * width * 4 + 4 * j];
				G[i * width + j] = imdata[i * width * 4 + 4 * j + 1];
				R[i * width + j] = imdata[i * width * 4 + 4 * j + 2];
			}
		}
		HalconCpp::GenImage3(&img, "byte", width, height, (Hlong)R, (Hlong)G, (Hlong)B);

		delete[] R;
		delete[] G;
		delete[] B;
		break;
	}//end switch
}


#endif // HAS_HALCON


} //end namespace mycv