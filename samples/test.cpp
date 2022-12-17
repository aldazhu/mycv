#include "mycv.hpp"

#include <string>
#include <iostream>


void test_error_code()
{
    int code = mycv::kImageEmpty;
    std::string msg = "test error";
    MYCV_ERROR(code, msg);

}

void test_integralImage()
{
    std::string src_path = "data\\target.jfif";
    cv::Mat source, result,opencv_sum,opencv_sqsum;
    source = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    mycv::integral(source, result);
    cv::integral(source, opencv_sum,opencv_sqsum, CV_32F,CV_32F);
    mycv::showImage(source,"source");
    mycv::showImage(opencv_sum,"opencv integral");
    mycv::showImage(result, "integral",0);

    //another integral image
    cv::Mat integral_image, integral_sq;
    mycv::integral(source, integral_image, integral_sq);

    
    
    mycv::showImage(integral_image, "integral image");
    mycv::showImage(integral_sq, "integral sq");
    mycv::showImage(opencv_sqsum, "opencv integral sq",0);


    //comapre two array
    cv::Mat diff_integral = integral_image - opencv_sum;
    cv::Mat diff_sqsum = integral_sq - opencv_sqsum;
    mycv::showImage(diff_integral, "diff integral");
    mycv::showImage(diff_sqsum, "diff sqsum",0);

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

void del()
{
    std::cout << DBL_MAX << std::endl;
    std::cout << FLT_MAX << std::endl;

}


int main()
{
    //test_NCC();
    test_integralImage();
    //del();
    system("pause");
    return 0;
}