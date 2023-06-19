/*****************************************************************//**
 * @file   threshold.cpp
 * @copyright 2023.  All right sreserved.
 * @brief  
 * 
 * @author zsP
 * @date   2023/6/16 
 *********************************************************************/
#include "threshold.h"
#include <immintrin.h>

namespace mycv {

int GetHist(const cv::Mat& src, int hist[256])
{
	if (src.empty())
	{
		MYCV_ERROR1(error_code::kImageEmpty);
		return error_code::kImageEmpty;
	}
	if (CV_8UC1 != src.type())
	{
		MYCV_ERROR1(error_code::kNotImplement);
		return error_code::kNotImplement;
	}

	// Init the hist
	for (int i = 0; i < 256; ++i)
		hist[i] = 0;

	for (int row = 0; row < src.rows; ++row)
	{
		const uchar* p = src.ptr<uchar>(row);
		for (int col = 0; col < src.cols; ++col)
		{
			hist[(int)( * (p + col))] += 1;
		}
	}

	return error_code::kSuccess;
}


int GetHistAVX(const cv::Mat& src, int hist[256])
{
	int histSize = 256;

	// 使用AVX指令集进行直方图计算
	__m256i zero = _mm256_setzero_si256();
	for (int i = 0; i < src.rows; ++i)
	{
		int j = 0;
		__m256i rowHist[32] = { zero };
		for (; j <= src.cols - 32; j += 32)  // 每次读取32个像素值，即一个AVX寄存器可容纳32个像素
		{
			__m256i pixels = _mm256_loadu_si256((__m256i*)(src.ptr(i) + j));  // 加载32个像素值
			__m256i indices = _mm256_set1_epi8(1);  // 设置每个像素值的权重为1
			rowHist[0] = _mm256_add_epi32(rowHist[0], _mm256_sad_epu8(pixels, zero));  // 对8个像素值的权重求和
			for (int k = 1; k < histSize / 8; ++k)  // 执行8次，用于把最终结果存储到rowHist数组中
			{
				pixels = _mm256_loadu_si256((__m256i*)(src.ptr(i) + j + k * 32));  // 再加载8个像素值
				indices = _mm256_add_epi8(indices, _mm256_set1_epi8(1));  // 每个像素值的权重仍为1
				rowHist[k] = _mm256_add_epi32(rowHist[k], _mm256_sad_epu8(pixels, zero));
			}
		}
		// 处理剩余的像素值
		for (; j < src.cols; ++j)
		{
			int val = src.at<uchar>(i, j);
			hist[val]++;
		}
		
	}

	return error_code::kSuccess;
}

int GetOtsuThresh(const cv::Mat& src)
{
	if (src.empty() || src.channels()!=1)
	{
		MYCV_ERROR2(error_code::kImageEmpty,"image is empty or channels() != 1");
		return -1;
	}
	
	// 
	int hist[256] = { 0 };
	if (error_code::kSuccess != GetHist(src, hist))
	{
		return -1;
	}

	float ep = FLT_MIN; // use to avoid devide by zero.
	float max_var = 0;
	int max_var_index = 0;
	for (int th = 1; th < 255; ++th)
	{
		// Calculate the mean value
		int sum1 = 0, sum2 = 0;
		int num1 = 0, num2 = 0;
		for (int i = 1; i <= th; ++i)
		{
			sum1 += hist[i] * i;
			num1 += hist[i];
		}
		float mean1 = sum1 / ((float)num1 + ep);;

		for (int i = th + 1; i < 255; ++i) //
		{
			sum2 += hist[i] * i;
			num2 += hist[i];
		}
		float mean2 = sum2 / ((float)num2 + ep);
		
		// the between-class variance
		float var = num1 * num2 * (mean1 - mean2) * (mean1 - mean2);
		if (var > max_var)
		{
			max_var = var;
			max_var_index = th;
		}
	}

	return max_var_index;
}


int OTSU(const cv::Mat& src, cv::Mat& dst, int mode)
{
	int th = GetOtsuThresh(src);
	if (-1 == th) return -1;

	int h = src.rows;
	int w = src.cols;
	dst = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
	for (int row = 0; row < h; ++row)
	{
		const uchar* p = src.ptr<uchar>(row);
		uchar* d = dst.ptr<uchar>(row);
		for (int col = 0; col < w; ++col)
		{
			switch (mode)
			{
			case 0:
				*(d + col) = *(p + col) > th ? 255 : 0;
				break;
			case 1:
				*(d + col) = *(p + col) > th ? *(p + col) : 0;
				break;
			case 2:
				*(d + col) = *(p + col) > th ? 0 : 255;
				break;
			default:
				*(d + col) = *(p + col) > th ? 255 : 0;
			}
		}
	}

	return th;
}

int OTSUAVX(const cv::Mat& src, cv::Mat& dst, int mode)
{
	int th = GetOtsuThresh(src);
	if (-1 == th) return -1;

	int h = src.rows;
	int w = src.cols;
	dst = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
	for (int row = 0; row < h; ++row)
	{
		const uchar* p = src.ptr<uchar>(row);
		uchar* d = dst.ptr<uchar>(row);
		for (int col = 0; col < w; ++col)
		{
			switch (mode)
			{
			case 0:
				*(d + col) = *(p + col) > th ? 255 : 0;
				break;
			case 1:
				*(d + col) = *(p + col) > th ? *(p + col) : 0;
				break;
			case 2:
				*(d + col) = *(p + col) > th ? 0 : 255;
				break;
			default:
				*(d + col) = *(p + col) > th ? 255 : 0;
			}
		}
	}

	return th;
}


}// end namespace mycv