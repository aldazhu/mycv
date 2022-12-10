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






    
} //end namespace mycv

#endif //!MYCV_NCC_H_