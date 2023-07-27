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
 * @brief ��ȡ�����䷽�����ֵ.
 * 
 * @param [in] src : CV_8UC1 ��ʽ��ͼƬ
 * @return  -1: input image empty
 * @return  others : threshold
 */
int GetOtsuThresh(const cv::Mat& src);

/**
* @brief OTSU�㷨��ʹ���ڷ�����С����䷽�����.
*
* @param [in] src : CV_8UC1 ��ʽ��ͼƬ
* @param [out] dst : ��ֵͼ
* @param [in] mode : 0: dstΪbinary��1��С��thresh��Ϊ0��2��binary INV��default is 0.
* @return  -1: input image empty
* @return  others : threshold
*/
int OTSU(const cv::Mat& src, cv::Mat& dst, int mode = 0);

/**
* @brief OTSU�㷨��ʹ���ڷ�����С����䷽�����.
*
* @param [in] src : CV_8UC1 ��ʽ��ͼƬ
* @param [out] dst : ��ֵͼ
* @param [in] mode : 0: dstΪbinary��1��С��thresh��Ϊ0��2��binary INV��default is 0.
* @return  -1: input image empty
* @return  others : threshold
*/
int OTSUAVX(const cv::Mat& src, cv::Mat& dst, int mode = 0);

/**
* @brief ��������ͼ��histֱ��ͼ.
*
* @param [in] src : CV_8UC1 ��ʽ��ͼƬ
* @param [out] hist : ����Ϊ256���������飬hist[i]��ֵ��ʾ�Ҷȼ�Ϊi�������ж��ٸ�
* @return  :
*/
int GetHist(const cv::Mat& src, int hist[256]);


/**
* @brief ��������ͼ��histֱ��ͼ.
*
* @param [in] src : CV_8UC1 ��ʽ��ͼƬ
* @param [out] hist : ����Ϊ256���������飬hist[i]��ֵ��ʾ�Ҷȼ�Ϊi�������ж��ٸ�
* @return  :
*/
int GetHistAVX(const cv::Mat& src, int hist[256]);

}// end namespace mycv
#endif// !MYCV_THRESHOLD_H_
