/**
 * @file NCC.cpp
 * @author WuMing (hello@hello.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-10
 * 
 * @copyright Copyright (c) 2022
 * @todo 彩色图（CV_8UC3）的的匹配，
 * @todo 输出得分结果图，
 * @todo 带mask的匹配，
 */

#include "NCC.h"
#include "utils.h"
#include "integral_image.h"

#include "stdio.h"
#include <iostream>

#include "opencv.hpp"

namespace mycv
{


/**
 * @brief 模板匹配，归一化交叉相关算法。衡量模板和待匹配图像的相似性时
 * 用(Pearson)相关系数来度量。
 * r=cov(X,Y)/(sigma(X) * sigma(Y))
 * 其中cov(X,Y): 表示两个变量的协方差
 * cov(X,Y) = E[(X-E(x)) * (Y-E(Y))] = E(XY) - E(x)E(Y)
 * sigma(X): 表示X变量的标准差
 * sigma(Y): 表示Y变量的标准差
 * 
 * @param source : 搜索图CV_8UC1格式
 * @param target ：模板图CV_8UC1格式
 * @param result : 匹配结果的map图
 * @return int : 程序运行的状态码
 */
int NormalizedCrossCorrelation(
    const cv::Mat &source,
    const cv::Mat &target,
    cv::Mat &result
    )
    {
        if(source.empty() || target.empty())
        {
            MYCV_ERROR2(kImageEmpty,"NCC empty input image");
            return kImageEmpty;
        }
        int H = source.rows;
        int W = source.cols;
        int t_h = target.rows;
        int t_w = target.cols;
        if(t_h > H || t_w > W)
        {
            MYCV_ERROR2(kBadSize,"NCC source image size should larger than targe image");
            return kBadSize;
        }

        //r = cov(X,Y)/(sigma(X) * sigma(Y))
        //sigma(X) = sqrt(var(X))
        int r_h = H - t_h + 1; //结果图的高度
        int r_w = W - t_w + 1;
        cv::Mat integral_image;//source的积分图
        cv::Mat sq_integral;//source 的像素平方的积分图
        integral(source,integral_image,sq_integral);

        const double target_size = (double)t_h * t_w;
        double target_mean = calculateMean(target);
        double target_var = calculateVariance(target, target_mean);
        double target_std_var = std::sqrt(target_var);
        result = cv::Mat::zeros(cv::Size(r_w, r_h), CV_32FC1);

        double region_sum = 0;
        double region_sq_sum = 0;
        for(int row = 0; row < r_h ; row++)
        {
            float * p = result.ptr<float>(row);
            for(int col = 0; col < r_w; col++)
            {
                cv::Rect ROI(col,row,t_w,t_h);//source上和目标图匹配的子图
                cv::Mat temp = source(ROI);
                //计算source中对应子块的均值
                getRegionSumFromIntegralImage(integral_image,col,row,col+t_w-1,row+t_h-1,region_sum);
                double temp_mean = region_sum / target_size;
                
                //计算两个图的协方差
                double cov = calculateCovariance(temp,target,temp_mean,target_mean);
                
                //计算source中对应子块的方差
                getRegionSumFromIntegralImage(sq_integral,col,row,col+t_w-1,row+t_h-1,region_sq_sum);

                double temp_var = (region_sq_sum - temp_mean*region_sum)/target_size;
                double temp_std_var = std::sqrt(temp_var);
                p[col] = cov / ((temp_std_var + 0.0000001) * (target_std_var + 0.0000001));
            }
        }


        return kSuccess;
    }


/**
 * @brief 模板匹配，归一化交叉相关算法。衡量模板和待匹配图像的相似性时
 * 用(Pearson)相关系数来度量。
 * r=cov(X,Y)/(sigma(X) * sigma(Y))
 * 其中cov(X,Y): 表示两个变量的协方差
 * cov(X,Y) = E[(X-E(x)) * (Y-E(Y))] = E(XY) - E(x)E(Y)
 * sigma(X): 表示X变量的标准差
 * sigma(Y): 表示Y变量的标准差
 *
 * @param source : 搜索图CV_8UC1格式
 * @param target ：模板图CV_8UC1格式
 * @param result : 匹配结果的map图
 * @return int : 程序运行的状态码
 */
int NormalizedCrossCorrelationFFT(
	const cv::Mat &source,
	const cv::Mat &target,
	cv::Mat &result
)
{
	if (source.empty() || target.empty())
	{
		MYCV_ERROR2(kImageEmpty, "NCC empty input image");
		return kImageEmpty;
	}
	int H = source.rows;
	int W = source.cols;
	int t_h = target.rows;
	int t_w = target.cols;
	if (t_h > H || t_w > W)
	{
		MYCV_ERROR2(kBadSize, "NCC source image size should larger than targe image");
		return kBadSize;
	}

	//r = cov(X,Y)/(sigma(X) * sigma(Y))
	//sigma(X) = sqrt(var(X))
	int r_h = H - t_h + 1; //结果图的高度
	int r_w = W - t_w + 1;
	cv::Mat integral_image;//source的积分图
	cv::Mat sq_integral;//source 的像素平方的积分图
	integral(source, integral_image, sq_integral);
	//cv::integral(source, integral_image, sq_integral, CV_64FC1, CV_64FC1);

	//计算模板图在source上的卷积
	cv::Mat conv;
	convFFT(source, target, conv);

	const double target_size = t_h * t_w;

	double target_mean = calculateMean(target);
	double target_var = calculateVariance(target, target_mean);
	double target_std_var = std::sqrt(target_var);
	result = cv::Mat::zeros(cv::Size(r_w, r_h), CV_32FC1);

	double region_sum = 0;
	double region_sq_sum = 0;
	for (int row = 0; row < r_h; row++)
	{
		float * p = result.ptr<float>(row);
		float * convp = conv.ptr<float>(row);
		for (int col = 0; col < r_w; col++)
		{
			cv::Rect ROI(col, row, t_w, t_h);//source上和目标图匹配的子图
			cv::Mat temp = source(ROI);
			//计算source中对应子块的均值
			getRegionSumFromIntegralImage(integral_image, col, row, col + t_w - 1, row + t_h - 1, region_sum);
			double temp_mean = region_sum / target_size;

			//计算两个图的协方差
			//cov(X,Y) = E(X*Y) - E(X)*E(Y)
			double cov = (*(convp + col)) / target_size - temp_mean * target_mean;
			//double cov = calculateCovariance(temp, target, temp_mean, target_mean);

			//计算source中对应子块的方差
			getRegionSumFromIntegralImage(sq_integral, col, row, col + t_w - 1, row + t_h - 1, region_sq_sum);

			double temp_var = (region_sq_sum - temp_mean * region_sum) / target_size;
			double temp_std_var = std::sqrt(temp_var);
			p[col] = cov / ((temp_std_var + 0.0000001) * (target_std_var + 0.0000001));
		}
	}


	return kSuccess;
}

/**
 * @brief 计算图像上ROI区域内的均值
 * 
 * @param input  : 输入的图像CV_8UC1
 * @param ROI  : 输入的ROI区域
 * @param mean  : 返回的区域均值
 * @return int 
 */
int calculateRegionMean(const cv::Mat &input,const cv::Rect &ROI,double &mean)
{
    if(input.empty())
    {
        MYCV_ERROR2(kImageEmpty,"input empty");
        return kImageEmpty;
    }
    if(1 != input.channels())
    {
        MYCV_ERROR2(kBadDepth,"Now only sopurt for one channel image");
        return kBadDepth;
    }
    int h = input.rows;
    int w = input.cols;
    
    if((ROI.x+ROI.width > w ) || (ROI.y+ROI.height > h)
    || ROI.width <= 0 || ROI.height <= 0 )
    {
        MYCV_ERROR2(kBadSize,"ROI is too big");
        return kBadSize;
    }
    int tpx = ROI.x;
    int tpy = ROI.y;
    int btx = ROI.x + ROI.width;
    int bty = ROI.y + ROI.height;
    double sum = 0;
    for(int row = tpy; row < bty; row++)
    {
        const uchar *p = input.ptr<uchar>(row);
        for (int col = tpx ; col < btx ; col++)
        {
            sum += p[col];
        }
    }
    int pixels_num = ROI.height * ROI.width;
    mean = sum / pixels_num;
    return kSuccess;
}

/**
 * @brief 计算两个输入图的协方差，两个输入图的尺寸需要一致,在计算目标图和原图子块的协方差时，
 * 目标图（模板图）是固定的，均值只需要计算一次，所以如果传入图像均值的话就不在计算均值，均值默认为-1
 * cov(X,Y): 表示两个变量的协方差
 * cov(X,Y) = E[ (X-E(x)) * (Y-E(Y)) ] = E(XY) - E(x)E(Y)
 * 
 * @param A  : 输入图A CV_8UC1
 * @param B  : 输入图B CV_8UC1
 * @param mean_a  : A的像素均值
 * @param mean_b  : B的像素均值
 * @return double : 两个图像的协方差
 */
double calculateCovariance(const cv::Mat &A, const cv::Mat &B,double mean_a,double mean_b)
{
    if(A.empty() || B.empty())
    {
        MYCV_ERROR2(kImageEmpty,"input image is empty");
        return kImageEmpty;
    }
    if (A.cols != B.cols || A.rows != B.rows)
    {
        MYCV_ERROR2(kBadSize,"mat A B should be in same size");
        return kBadSize;
    }
    
    //E(XY)
    double sum = 0;
    for (int row = 0; row < A.rows; row++)
    {
        const uchar *pa = A.ptr<uchar>(row);
        const uchar *pb = B.ptr<uchar>(row);
        for (int  col = 0; col < A.cols; col++)
        {
            sum += (double)pa[col] * (double)pb[col];
        }
        
    }

    double mean_AB = sum / ((double)A.rows * (double)A.cols);

    if (-1 == mean_a)
    {
        mean_a = calculateMean(A);
    }
    if (-1 == mean_b)
    {
        mean_b = calculateMean(B);
    }
    
    //cov(X,Y) = E[ (X-E(x)) * (Y-E(Y)) ] = E(XY) - E(x)E(Y)
    double cov_AB = mean_AB - (mean_a * mean_b);
    
    return cov_AB;
}

/**
 * @brief 计算输入图像的方差，如果已知mean就不再计算mean
 * 
 * @param image  : 输入图CV_8UC1
 * @param mean  : 图像的灰度均值，默认值为-1，不输入时会计算mean
 * @return double ：图像的方差
 */
double calculateVariance(const cv::Mat &image,double mean)
{
    if (image.empty())  
    {
        MYCV_ERROR2(kImageEmpty,"empty image");
        return -1;//正常的方差不会小于0
    }
    if (-1 == mean)
    {
        mean = calculateMean(image);
    }

    double sum = 0 ;
    for (int  row = 0; row < image.rows; row++)
    {
        const uchar * p = image.ptr<uchar>(row);
        for (int col = 0; col < image.cols; col++)
        {
            sum += (p[col] - mean) * (p[col] - mean);
        }
        
    }

    double var = sum / ((double)image.cols * (double)image.rows);
    
    return var;    
}



/**
 * @brief 计算输入图的灰度均值
 * 
 * @param image  : 输入图CV_8UC1
 * @return double ： 输入图像的灰度均值
 */
double calculateMean(const cv::Mat &image)
{
     if (image.empty())  
    {
        MYCV_ERROR2(kImageEmpty,"empty image");
        return -1;
    }

    double sum = 0 ;
    for (int  row = 0; row < image.rows; row++)
    {
        const uchar * p = image.ptr<uchar>(row);
        for (int col = 0; col < image.cols; col++)
        {
            sum += p[col];
        }
        
    }

    double mean = sum / ((double)image.cols * (double)image.rows);
    return mean;
}


/**
 * @brief 用FFT实现两个图像的卷积，src: H*W ,kernel: h*w,
 * 把卷积的过程想象成kernel在src上滑动，记在src上和kernel对应的子图像块为Sxy,
 * conv(x,y) = \sigma \sigma Sxy(i,j)*kernel(i,j),i in [0,h) ,j in [0,w)
 * output: (H-h+1)*(W-w+1)
 * 
 * @param src  : CV_8UC1的图像
 * @param kernel  : CV_8UC1的图像
 * @param output  : CV_32FC1的图像
 * @return int 
 */
int convFFT(const cv::Mat &src, const cv::Mat &kernel, cv::Mat& output)
{
	if (src.empty() || kernel.empty())
	{
        MYCV_ERROR2(kImageEmpty,"input is empty");
		return kImageEmpty;
	}
	int dft_h = cv::getOptimalDFTSize(src.rows + kernel.rows - 1);
	int dft_w = cv::getOptimalDFTSize(src.cols + kernel.cols - 1);

	cv::Mat dft_src = cv::Mat::zeros(dft_h, dft_w, CV_32F);
	cv::Mat dft_kernel = cv::Mat::zeros(dft_h, dft_w, CV_32F);

	cv::Mat dft_src_part = dft_src(cv::Rect(0, 0, src.cols, src.rows));
	cv::Mat dft_kernel_part = dft_kernel(cv::Rect(0, 0, kernel.cols, kernel.rows));

	//把src,kernel分别拷贝放到dft_src,dft_kernel左上角
	src.convertTo(dft_src_part, dft_src_part.type());
	kernel.convertTo(dft_kernel_part, dft_kernel_part.type());

	cv::dft(dft_src, dft_src, 0, dft_src.rows);
	cv::dft(dft_kernel, dft_kernel, 0, dft_kernel.rows);

	cv::mulSpectrums(dft_src, dft_kernel, dft_src, 0, true);
	
	int output_h = abs(src.rows - kernel.rows) + 1;
	int output_w = abs(src.cols - kernel.cols) + 1;
	cv::dft(dft_src, dft_src, cv::DFT_INVERSE + cv::DFT_SCALE, output_h);;

	cv::Mat corr = dft_src(cv::Rect(0, 0, output_w,output_h));

	output = corr;

    return kSuccess;
}


} //end namespace mycv
