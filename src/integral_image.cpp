/**
 * @file integral_image.cpp
 * @author WuMing (hello@hello.com)
 * @brief 计算积分图
 * @version 0.1
 * @date 2022-12-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "integral_image.h"

#include "mycv.hpp"

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
int integral(const cv::Mat &image,cv::Mat &integral_image)
{
    if(image.empty())
    {
        MYCV_ERROR(kImageEmpty,"image empty");
        return kImageEmpty;
    }

    int h = image.rows;
    int w = image.cols;
    integral_image = cv::Mat::zeros(cv::Size(w+1,h+1),CV_32FC1);

    //SAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)
    for(int i = 0; i < h ; i++)
    {
        const uchar *ps = image.ptr<uchar>(i);
        float *pd_m1 = integral_image.ptr<float>(i);//integral 的"上一行"
        float *pd = integral_image.ptr<float>(i+1); //integral 的"当前行"
        for(int j = 0; j < w; j++)
        {
            pd[j+1] = pd[j] + pd_m1[j+1] - pd_m1[j] + ps[j];
        }
    }

    return kSuccess;
}


/**
 * @brief 计算输入图的积分图,为了提高计算效率，可以让积分图比输入图多一行一列，
 * 具体的就是在原图左边插入一列0，上面插入一行0，设原图为I，积分图为SAT(sum area of table)
 * 则：SAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)
 * 这样就不用考虑下边界的情况，省掉很多判断条件
 * 
 * @param image  : 输入图CV_8UC1，MxN
 * @param integral_image  : 积分图CV_32FC1,(M+1)x(N+1)
 * @param integral_sq : 平方的积分图CV_32FC1,(M+1)x(N+1)
 * @return int 
 */
int integral(const cv::Mat &image,cv::Mat &integral_image,cv::Mat &integral_sq)
{
     if(image.empty())
    {
        MYCV_ERROR(kImageEmpty,"image empty");
        return kImageEmpty;
    }

    int h = image.rows;
    int w = image.cols;
    integral_image = cv::Mat::zeros(cv::Size(w+1,h+1),CV_32FC1);
    integral_sq = cv::Mat::zeros(cv::Size(w+1,h+1),CV_32FC1);

    //SAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)
    //SQAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)*I(i,j)
    for(int i = 0; i < h ; i++)
    {
        const uchar *ps = image.ptr<uchar>(i);
        float *pd_m1 = integral_image.ptr<float>(i);//integral 的"上一行"
        float *pd = integral_image.ptr<float>(i+1); //integral 的"当前行"
        float *pqd_m1 = integral_sq.ptr<float>(i);
        float *pqd = integral_sq.ptr<float>(i+1);
        for(int j = 0; j < w; j++)
        {
            pd[j+1] = pd[j] + pd_m1[j+1] - pd_m1[j] + (float)ps[j];
            pqd[j+1] = pqd[j] + pqd_m1[j+1] - pqd_m1[j] + (float)ps[j] * (float)ps[j];
        }
    }


    return kSuccess;
}



}//end namespace mycv