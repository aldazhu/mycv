/*****************************************************************//**
 * @file   threshold.h
 * @copyright 2023. 
 * @brief  
 * 
 * @author zsP
 * @date   2023/6/16 
 *********************************************************************/

#ifndef MYCV_THRESHOLD_H_
#define MYCV_THRESHOLD_H_

#include "mycv.hpp"

namespace mycv {

/**
 * @brief 获取最大类间方差的阈值.
 * 
 * @param [in] src : CV_8UC1 格式的图片
 * @return  -1: input image empty
 * @return  others : threshold
 */
int GetOtsuThresh(const cv::Mat& src);

/**
* @brief OTSU算法，使类内方差最小，类间方差最大.
*
* @param [in] src : CV_8UC1 格式的图片
* @param [out] dst : 二值图
* @param [in] mode : 0: dst为binary，1：小于thresh的为0，2：binary INV，default is 0.
* @return  -1: input image empty
* @return  others : threshold
*/
int OTSU(const cv::Mat& src, cv::Mat& dst, int mode = 0);

/**
* @brief OTSU算法，使类内方差最小，类间方差最大.
*
* @param [in] src : CV_8UC1 格式的图片
* @param [out] dst : 二值图
* @param [in] mode : 0: dst为binary，1：小于thresh的为0，2：binary INV，default is 0.
* @return  -1: input image empty
* @return  others : threshold
*/
int OTSUAVX(const cv::Mat& src, cv::Mat& dst, int mode = 0);

/**
* @brief 计算输入图的hist直方图.
*
* @param [in] src : CV_8UC1 格式的图片
* @param [out] hist : 长度为256的整数数组，hist[i]的值表示灰度级为i的像素有多少个
* @return  :
*/
int GetHist(const cv::Mat& src, int hist[256]);


/**
* @brief 计算输入图的hist直方图.
*
* @param [in] src : CV_8UC1 格式的图片
* @param [out] hist : 长度为256的整数数组，hist[i]的值表示灰度级为i的像素有多少个
* @return  :
*/
int GetHistAVX(const cv::Mat& src, int hist[256]);

}// end namespace mycv
#endif// !MYCV_THRESHOLD_H_
