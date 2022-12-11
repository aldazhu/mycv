#include "utils.h"

#include "stdio.h"

namespace mycv
{



/**
 * @brief 把错误码翻译为文字
 * 
 * @param error_code  : 
 * @return const char* 
 */
const char* error_code_string(int error_code)
{
    char buf[256] = {};
    switch(error_code)
    {
        case kImageEmpty:   return "Image is empty!";
        case kOutOfRange:   return "Index out of range!";
    }

    return "Unknown error";
}

void error(
    int error_code, 
    const std::string &error_msg, 
    const char* func_name, 
    const char* source_file, 
    int code_line
    )
    {
        printf("%s : %s. in %s function: %s , line : %d",
        error_code_string(error_code),error_msg.c_str(),source_file,func_name,code_line);
    }
}// end namespace mycv