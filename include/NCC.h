/**
 * @file NCC.h
 * @author WuMing (hello@hello.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MYCV_NCC_H_
#define MYCV_NCC_H_

#include "opencv.hpp"

namespace mycv{

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
    );



/**
 * @brief 计算图像上ROI区域内的均值
 * 
 * @param input  : 输入的图像CV_8UC1
 * @param ROI  : 输入的ROI区域
 * @param mean  : 返回的区域均值
 * @return int 状态码，用于表示函数运行状态
 */
int calculateRegionMean(const cv::Mat &input,const cv::Rect &ROI,double &mean);


/**
 * @brief 计算两个输入图的协方差，两个输入图的尺寸需要一致,在计算目标图和原图子块的协方差时，
 * 目标图（模板图）是固定的，均值只需要计算一次，所以如果传入图像均值的话就不在计算均值，均值默认为-1
 * cov(X,Y): 表示两个变量的协方差
 * cov(X,Y) = E[ (X-E(x)) * (Y-E(Y)) ] = E(XY) - E(x)E(Y)
 * 
 * @param A  : 输入图A CV_8UC1
 * @param B  : 输入图B CV_8UC1
 * @param mean_a  : A的灰度均值，默认值为-1，不输入时会计算mean
 * @param mean_b  : B的灰度均值，默认值为-1，不输入时会计算mean
 * @return double : 两个图像的协方差
 */
double covariance(const cv::Mat &A, const cv::Mat &B,double mean_a=-1,double mean_b=-1);
    
/**
 * @brief 计算输入图像的方差，如果已知mean就不再计算mean
 * 
 * @param image  : 输入图CV_8UC1
 * @param mean  : 图像的灰度均值，默认值为-1，不输入时会计算mean
 * @return double ：图像的方差
 */
double variance(const cv::Mat &image,double mean=-1);


/**
 * @brief 计算输入图的灰度均值
 * 
 * @param image  : 输入图CV_8UC1
 * @return double ： 输入图像的灰度均值
 */
double mean(const cv::Mat &image);


} //end namespace mycv

#endif //!MYCV_NCC_H_