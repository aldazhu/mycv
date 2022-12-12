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



}//end namespace mycv

#endif //!MYCV_UTILS_H_