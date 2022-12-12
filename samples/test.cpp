#include "mycv.hpp"

#include <string>
#include <iostream>

void test_error_code()
{
    int code = mycv::kImageEmpty;
    std::string msg = "test error";
    MYCV_ERROR(code, msg);

}

void test_NCC()
{
    std::string src_path = "data\\source.jfif";
    std::string target_path = "data\\target.jfif";
    cv::Mat source, target,result;
    source = cv::imread(src_path,cv::IMREAD_GRAYSCALE);
    target = cv::imread(target_path, cv::IMREAD_GRAYSCALE);

    mycv::NormalizedCrossCorrelation(source, target, result);
    mycv::showImage(target, "target");
    mycv::showImage(source, "source");
    mycv::showImage(result, "result", 0);

    result += 1;
    result *= 128;
    result.convertTo(result, CV_8U);
    cv::imwrite("data\\result.png", result);
}




int main()
{
    test_NCC();
    return 0;
}