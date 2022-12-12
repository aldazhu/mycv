#include "utils.h"

#include "stdio.h"
#include "opencv.hpp"

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
        case kSuccess:      return "Success!";
        case kImageEmpty:   return "Image is empty!";
        case kOutOfRange:   return "Index out of range!";
        case kBadSize:      return "Bad size!";
        case kBadDepth:     return "bad depth!";
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
        //
        printf("In %s , %d line ,error: %s in function %s , %s \n",
        source_file,code_line,error_code_string(error_code),func_name,error_msg.c_str());
    }


void showImage(const cv::Mat& image, const std::string& name, int waitMode,int windowMode)
{
    if (image.empty())
    {
        MYCV_ERROR(kImageEmpty, "image is empty!");
        return;
    }
    cv::namedWindow(name, windowMode);
    cv::imshow(name, image);
    cv::waitKey(waitMode);
}








}// end namespace mycv