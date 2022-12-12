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
 * 则：SAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)
 * 这样就不用考虑下边界的情况，省掉很多判断条件
 * 
 * @param image  : 输入图CV_8UC1，MxN
 * @param integral_image  : 积分图CV_32FC1,(M+1)x(N+1)
 * @return int 
 */
int integral(const cv::Mat &image,cv::Mat &integral_image);

}// end namespace mycv



#endif //!MYCV_INTEGRAL_IMAGE_H_