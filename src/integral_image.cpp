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
    integral_image = cv::Mat::zeros(cv::Size(w+1,h+1),CV_64FC1);

    //SAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)
    for(int i = 0; i < h ; i++)
    {
        const uchar *ps = image.ptr<uchar>(i);
        double *pd_m1 = integral_image.ptr<double>(i);//integral 的"上一行"
        double *pd = integral_image.ptr<double>(i+1); //integral 的"当前行"
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
    integral_image = cv::Mat::zeros(cv::Size(w+1,h+1),CV_64FC1);
    integral_sq = cv::Mat::zeros(cv::Size(w+1,h+1),CV_64FC1);

    //SAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)
    //SQAT(i,j)=S(i,j-1)+S(i-1,j)-S(i-1,j-1)+I(i,j)*I(i,j)
    for(int i = 0; i < h ; i++)
    {
        const uchar *ps = image.ptr<uchar>(i);
        double *pd_m1 = integral_image.ptr<double>(i);//integral 的"上一行"
        double *pd = integral_image.ptr<double>(i+1); //integral 的"当前行"
        double *pqd_m1 = integral_sq.ptr<double>(i);
        double *pqd = integral_sq.ptr<double>(i+1);
        for(int j = 0; j < w; j++)
        {
            pd[j+1] = pd[j] + pd_m1[j+1] - pd_m1[j] + (double)ps[j];
            pqd[j+1] = pqd[j] + pqd_m1[j+1] - pqd_m1[j] + (double)ps[j] * (double)ps[j];
        }
    }


    return kSuccess;
}


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
int getRegionSumFromIntegralImage(const cv::Mat & integral,int tpx,int tpy,int btx,int bty,double &sum)
{
    if(integral.empty())
    {
        MYCV_ERROR(mycv::kImageEmpty,"Input image is empty!");
        return mycv::kImageEmpty;
    }
    if(tpx > btx 
    || tpy > bty
    || tpx < 0 
    || tpy <0
    || btx > integral.cols - 1
    || bty > integral.rows - 1)
    {
        MYCV_ERROR(mycv::kBadSize,"0 <= tpx <= btx <= w, && 0<= tpy <= bty <= h");
        return mycv::kBadSize;
    }
    const double *ptp = integral.ptr<double>(tpy+1);
    const double *pbt = integral.ptr<double>(bty+1);
    
    sum = (*(pbt+btx+1)) - (*(pbt+tpx+1)) - (*(ptp+btx+1)) + (*(ptp+tpx+1)); 

    return mycv::kSuccess;
}


}//end namespace mycv