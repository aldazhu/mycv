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
 * @param loc : 匹配位置
 * @param max_value ：匹配得分
 * @return int 
 */
int NormalizedCrossCorrelation(
    const cv::Mat &source,
    const cv::Mat &target,
    cv::Point &loc,
    float &max_value
    )
    {
        if(source.empty() || target.empty())
        {
            MYCV_ERROR(kImageEmpty,"NCC empty input image");
            return kImageEmpty;
        }
        int H = source.rows;
        int W = source.cols;
        int t_h = target.rows;
        int t_w = target.cols;
        if(t_h > H || t_w > W)
        {
            MYCV_ERROR(kBadSize,"NCC source image size should larger than targe image");
            return kBadSize;
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
        MYCV_ERROR(kImageEmpty,"input empty");
        return kImageEmpty;
    }
    if(1 != input.channels())
    {
        MYCV_ERROR(kBadDepth,"Now only sopurt for one channel image");
        return kBadDepth;
    }
    int h = input.rows;
    int w = input.cols;
    
    if((ROI.x+ROI.width > w ) || (ROI.y+ROI.height > h)
    || ROI.width <= 0 || ROI.height <= 0 )
    {
        MYCV_ERROR(kBadSize,"ROI is too big");
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
double covariance(const cv::Mat &A, const cv::Mat &B,double mean_a,double mean_b)
{
    if(A.empty() || B.empty())
    {
        MYCV_ERROR(kImageEmpty,"input image is empty");
        return kImageEmpty;
    }
    if (A.cols != B.cols || A.rows != B.rows)
    {
        MYCV_ERROR(kBadSize,"mat A B should be in same size");
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
            sum += pa[col] * pb[col];
        }
        
    }

    double mean_AB = sum / (A.rows * A.cols);

    cv::Rect ROI(0,0,A.cols,A.rows);
    if (mean_a == -1)
    {
        calculateRegionMean(A,ROI,mean_a);
    }
    if (mean_b == -1)
    {
        calculateRegionMean(B,ROI,mean_b);
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
double variance(const cv::Mat &image,double mean)
{
    if (image.empty())  
    {
        MYCV_ERROR(kImageEmpty,"empty image");
        return -1;//正常的方差不会小于0
    }
    if (-1 == mean)
    {
        mean = mean(image);
    }

    double sum = 0 ;
    for (int  row = 0; row < image.cols; row++)
    {
        const uchar * p = image.ptr<uchar>(row);
        for (int col = 0; col < image.cols; col++)
        {
            sum += (p[col] - mean) * (p[col] - mean);
        }
        
    }

    double var = sum / (image.cols * image.rows);
    
    return var;    
}



/**
 * @brief 计算输入图的灰度均值
 * 
 * @param image  : 输入图CV_8UC1
 * @return double ： 输入图像的灰度均值
 */
double mean(const cv::Mat &image)
{
     if (image.empty())  
    {
        MYCV_ERROR(kImageEmpty,"empty image");
        return -1;
    }

    double sum = 0 ;
    for (int  row = 0; row < image.cols; row++)
    {
        const uchar * p = image.ptr<uchar>(row);
        for (int col = 0; col < image.cols; col++)
        {
            sum += p[col];
        }
        
    }

    double mean = sum / (image.cols * image.rows);
    return mean;
}




} //end namespace mycv
