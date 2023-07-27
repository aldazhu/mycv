#ifndef MYCV_RESIZE_H_
#define MYCV_RESIZE_H_

#include "mycv.hpp"


namespace mycv
{
	/**
	 * @brief .
	 * 
	 * @param [] src : 
	 * @param [] dst : 
	 * @param [] new_h : 
	 * @param [] new_w : 
	 * @param [] mode : 
	 * @return  :
	 */
	int Resize(const cv::Mat& src, cv::Mat& dst, const int new_h, const int new_w, const int mode);
}


#endif // MYCV_RESIZE_H_
