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

namespace mycv
{

enum error_code{
    kImageEmpty,
    kOutOfRange
};

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

}//end namespace mycv

#endif //!MYCV_UTILS_H_