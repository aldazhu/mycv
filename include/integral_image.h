/**
 * @file integral_image.h
 * @author WuMing (hello@hello.com)
 * @brief 计算积分图
 * @version 0.1
 * @date 2022-12-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MYCV_INTEGRAL_IMAGE_H_
#define MYCV_INTEGRAL_IMAGE_H_

#include "opencv.hpp"

namespace mycv
{

/**
 * @brief 计算输入图的积分图,为了提高计算效率，可以让积分图比输入图多一行一列，
 * 具体的就是在原图左边插入一列0，上面插入一行0，设原图为I，积分图为SAT(sum area of table)
 * 则：SAT(i,j) = SAT(i,j-1) + SAT(i-1,j) - SAT(i-1,j-1) + I(i,j)
 * 这样就不用考虑下边界的情况，省掉很多判断条件
 * 
 * @param image  : 输入图CV_8UC1，MxN
 * @param integral_image  : 积分图CV_32FC1,(M+1)x(N+1)
 * @return int 
 */
int integral(const cv::Mat &image,cv::Mat &integral_image);

/**
 * @brief 计算输入图的积分图,为了提高计算效率，可以让积分图比输入图多一行一列，
 * 具体的就是在原图左边插入一列0，上面插入一行0，设原图为I，积分图为SAT(summed area table)
 * 则：SAT(i,j) = SAT(i,j-1) + SAT(i-1,j) - SAT(i-1,j-1) + I(i,j)
 * SQAT(i,j) = SQAT(i,j-1) + SQAT(i-1,j) - SQAT(i-1,j-1) + I(i,j) * I(i,j)
 * 这样就不用考虑下边界的情况，省掉很多判断条件
 * 
 * @param image  : 输入图CV_8UC1，MxN
 * @param integral_image  : 积分图CV_32FC1,(M+1)x(N+1)
 * @param integral_sq : 平方的积分图CV_32FC1,(M+1)x(N+1)
 * @return int 
 */
int integral(const cv::Mat &image,cv::Mat &integral_image,cv::Mat &integral_sq);

int integralIPP(const cv::Mat &image, cv::Mat &integral_image, cv::Mat &integral_sq);

/**
 * @brief Get the Region sum From Integral Image or sq integral image
 * 原图上的区域为tpx，tpy,btx,bty,在积分图或者平方的积分图上的位置为tpx+1,tpy+1,btx+1,bty+1
 * region sum = SAT(btx+1,bty+1) - SAT(tpx+1,bty+1) - SAT(btx+1,tpy+1) + SAT(tpx+1,tpy+1)
 * 
 * @param integral  : 像素和的积分图或者像素平方的积分图，CV_64FC1格式
 * @param tpx  : x of top left
 * @param tpy  : y of top right
 * @param btx  : x of bottom right
 * @param bty  : y of bottom right
 * @param sum  : 区域和
 * @return int : 程序运行状态码
 */
int getRegionSumFromIntegralImage(const cv::Mat & integral,int tpx,int tpy,int btx,int bty,double &sum);

}// end namespace mycv



#endif //!MYCV_INTEGRAL_IMAGE_H_