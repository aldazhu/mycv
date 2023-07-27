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

	// ʹ��AVXָ�����ֱ��ͼ����
	__m256i zero = _mm256_setzero_si256();
	for (int i = 0; i < src.rows; ++i)
	{
		int j = 0;
		__m256i rowHist[32] = { zero };
		for (; j <= src.cols - 32; j += 32)  // ÿ�ζ�ȡ32������ֵ����һ��AVX�Ĵ���������32������
		{
			__m256i pixels = _mm256_loadu_si256((__m256i*)(src.ptr(i) + j));  // ����32������ֵ
			__m256i indices = _mm256_set1_epi8(1);  // ����ÿ������ֵ��Ȩ��Ϊ1
			rowHist[0] = _mm256_add_epi32(rowHist[0], _mm256_sad_epu8(pixels, zero));  // ��8������ֵ��Ȩ�����
			for (int k = 1; k < histSize / 8; ++k)  // ִ��8�Σ����ڰ����ս���洢��rowHist������
			{
				pixels = _mm256_loadu_si256((__m256i*)(src.ptr(i) + j + k * 32));  // �ټ���8������ֵ
				indices = _mm256_add_epi8(indices, _mm256_set1_epi8(1));  // ÿ������ֵ��Ȩ����Ϊ1
				rowHist[k] = _mm256_add_epi32(rowHist[k], _mm256_sad_epu8(pixels, zero));
			}
		}
		// ����ʣ�������ֵ
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
	float ep = 0.0000000001f; //  avoid devide by zero.
	int hist[256] = { 0 };
	if (error_code::kSuccess != GetHist(src, hist))
	{
		return -1;
	}
	
	int sum = 0;// sum(i * hist[i])
	int num = 0;// the number of pixels
	for (int i = 0; i < 256; ++i)
	{
		sum += i * hist[i];
		num += hist[i];
	}

	float max_var = -1.0;
	int max_var_index = 0;
	int sum1 = 0; // ԭ��Ӧ���� 0*hist[0],ֵΪ0
	int num1 = hist[0];
	for (int th = 1; th < 255; ++th) // th in [1,254]
	{
		// Calculate the mean value
		
		sum1 +=  th * hist[th]; // �ۼƵ�һ������������ÿ�δ�ͷ��ʼ���, (����float�����Ӻܶ�ʱ��)
		num1 += hist[th];
		float mean1 = sum1 / (num1 + ep);
		int num2 = num - num1;
		float mean2 = (sum - sum1) / (num - num1 + ep);
		
		// the between-class variance
		// ���������������num�ᵼ�¾�����ʧ��ǰ���num1*num2�ܴ󣬺������Ҫ����ܶ�λС��������˵�ʱ��С�����ڸ��ˡ�
		// ERROR!: float var = num1 * num2  * (mean1 - mean2) * (mean1 - mean2);// �����������num�Ƕ���ģ���ΪֻҪvar���Ϳ����ˣ����ǲ�������ʧ����
		float var = ((float)num1 / num) * ((float)num2 / num) * (mean1 - mean2) * (mean1 - mean2);
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