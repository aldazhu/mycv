/**
 * @file utils.h
 * @author WuMing (hello@hello.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MYCV_UTILS_H_
#define MYCV_UTILS_H_

#include "mycv_def.h"

#include <string>

#include "opencv.hpp"

namespace mycv
{

/**
 * @brief error code
 * 
 */
enum error_code{
    kSuccess = 0,
    kImageEmpty,
    kOutOfRange,
    kBadSize,
    kBadDepth,
    kBadInput,
	kNotImplement,
	kDoseNotSupportImageType,
};

/**
 * @brief 把错误码翻译为文字
 * 
 * @param error_code  : 
 * @return const char* 
 */
const char* error_code_string(int error_code);

/**
 * @brief 用于输出错误信息
 * 
 * @param error_code  : 错误类别
 * @param error_msg  : 错误信息
 * @param func_name  : 函数名
 * @param source_file  : 源文件名
 * @param code_line  : 错误处在源文件中的位置
 */
void error(
    int error_code, 
    const std::string &error_msg, 
    const char* func_name, 
    const char* source_file, 
    int code_line
    );


void showImage(const cv::Mat& image, const std::string& name,int waitMode=1, int windowMode = 0);


/**
 * @brief 比较两个数组的差异，返回两个数组的差异的绝对值之和
 * 
 * @tparam T : 数组类型
 * @param A  : 数组A
 * @param B  : 数组B
 * @param n  : 比较前n个元素
 * @return double : 返回两个数组前n个元素差异绝对值之和
 */
template<typename TA,typename TB>
double compareArray(TA* A, TB* B,int n)
{
    if(nullptr == A || nullptr == B)
    {
        MYCV_ERROR2(mycv::kBadInput,"input array is none");
        throw "input array is none!";
    }

    double sum = 0;
    for(int i=0; i < n; i++)
    {
        sum += (A[i] - B[i]) * (A[i] - B[i]);
    }

    return sum;
}


/**
 * @brief 计时工具.
 */
template <typename Time_T>
class Timer
{
public:

	Timer()
	{
		start = std::chrono::steady_clock::now();
	}

	void Restart()
	{
		start = std::chrono::steady_clock::now();
	}

	double Duration()
	{
		auto end = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<Time_T>(end - start);
		//std::cout << "Run " << duration << "\n" << std::flush;;// C++ 20 重载了operator<<

		std::cout << "Run " << duration.count();
		
		if (std::is_same<std::chrono::duration<double, std::micro>, Time_T>()) {
			std::cout << " us\n";
		}
		else if (std::is_same<std::chrono::duration<double, std::milli>, Time_T>()) {
			std::cout << " ms\n";
		}
		else if (std::is_same<std::chrono::duration<double, std::nano>, Time_T>()) {
			std::cout << " ns\n";
		}

		return duration.count();
	}

	~Timer() = default;

	std::chrono::steady_clock::time_point start;

};

using Timer_ms = Timer<std::chrono::duration<double, std::milli>>;
using Timer_us = Timer<std::chrono::duration<double, std::micro>>;
using Timer_ns = Timer<std::chrono::duration<double, std::nano>>;


inline int CheckImageEmpty(const cv::Mat& image)
{
	if (image.empty()) return mycv::error_code::kImageEmpty;
	return mycv::error_code::kSuccess;
}

}//end namespace mycv

#endif //!MYCV_UTILS_H_