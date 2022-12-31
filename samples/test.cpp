#include "mycv.hpp"

#include <string>
#include <iostream>

#include "spdlog/logger.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

void cmp_speed()
{
    const int TIMES = 1000;

    std::chrono::steady_clock::time_point start_time, end_time;
    double abs_runtime = 0, integral_runtime = 0;
    for (int size = 100; size < 1000; size += 100)
    {
        cv::Mat target = cv::Mat(cv::Size(size, size), CV_8UC1);
        cv::randu(target, cv::Scalar(0), cv::Scalar(255));
        spdlog::info("\n \n image size(h, w) = ({}, {})",target.rows,target.cols);
        int t_h = target.rows;
        int t_w = target.cols;
        const double target_size = (double)t_h * t_w;
        double target_region_sum, target_region_sqsum, target_mean, target_var, target_std_var;

        cv::Mat target_sum, target_sqsum;
        start_time = std::chrono::steady_clock::now();;
        for (int times = 0; times < TIMES; times++)
        {
            //mycv::integral(target, target_sum, target_sqsum);
            cv::integral(target, target_sum, target_sqsum,CV_64F,CV_64F);
            
            mycv::getRegionSumFromIntegralImage(target_sum, 0, 0, target.cols - 1, target.rows - 1, target_region_sum);
            mycv::getRegionSumFromIntegralImage(target_sqsum, 0, 0, target.cols - 1, target.rows - 1, target_region_sqsum);
            target_mean = target_region_sum / target_size;
            target_var = (target_region_sqsum - target_mean * target_region_sum) / target_size;
            target_std_var = std::sqrt(target_var);
        }
        end_time = std::chrono::steady_clock::now();

        integral_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() ;
        spdlog::info("积分图的方法计算target的均值和方差");
        spdlog::info("run {0} times,  use {1}ms", TIMES, integral_runtime);
        spdlog::info("mean:{}, std variance:{}", target_mean, target_std_var);
        
        cv::Mat mean_mat, stddev_mat;
        start_time = std::chrono::steady_clock::now();;
        for (int times = 0; times < TIMES; times++)
        {
            //target_mean = mycv::calculateMean(target);
            //target_var = mycv::calculateVariance(target, target_mean);
            //target_std_var = std::sqrt(target_var);
            cv::meanStdDev(target, mean_mat, stddev_mat);

        }
        end_time = std::chrono::steady_clock::now();

        abs_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() ;
        spdlog::info("直接计算target的均值和方差");
        spdlog::info("run {0} times, use {1}ms", TIMES, abs_runtime);
        //spdlog::info("mean:{}, std variance:{}", target_mean, target_std_var);
        spdlog::info("opencv mean:{}, std variance:{}", mean_mat.at<double>(0), stddev_mat.at<double>(0));
        
        spdlog::info("abs_runtime / integral_image = {}",abs_runtime/integral_runtime);

    }
    
}

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
    cv::integral(source, opencv_sum,opencv_sqsum, CV_64F,CV_64F);
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
    
    std::cout << "integral difference sum" << cv::sum(diff_integral) << std::endl;;
    std::cout << "integral_sq difference sum" << cv::sum(diff_integral) << std::endl;;

    mycv::showImage(diff_integral, "diff integral");
    mycv::showImage(diff_sqsum, "diff sqsum",0);
    
    //save image
    cv::normalize(integral_image, integral_image, 1.0, 0.0, cv::NORM_MINMAX);
    integral_image.convertTo(integral_image, CV_8U, 255, 0);
    cv::normalize(integral_sq, integral_sq, 1.0, 0.0, cv::NORM_MINMAX);
    integral_sq.convertTo(integral_sq, CV_8U, 255, 0);
    cv::imwrite("data\\integral.png", integral_image);
    cv::imwrite("data\\integral_sq.png", integral_sq);


}

void test_NCC_speed()
{
    const int TIMES = 1;
    std::string src_path = "data\\source.jfif";
    std::string target_path = "data\\target.jfif";
    std::string log_path = "ncc_speed.txt";
    cv::Mat source, target, result;
    //source = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    //target = cv::imread(target_path, cv::IMREAD_GRAYSCALE);
    std::chrono::steady_clock::time_point start_time,end_time;
    double myncc_runtime = 0, opencv_runtime = 0;

    auto logger = spdlog::basic_logger_mt("NCC", log_path);
    logger->set_level(spdlog::level::critical);
    // location
    double min_value, max_value;
    cv::Point min_loc, max_loc;
    for (int src_size = 500; src_size <= 1200; src_size += 100)
    {
        source = cv::Mat(cv::Size(src_size, src_size), CV_8UC1);
        target = cv::Mat(cv::Size(100, 100), CV_8UC1);
        cv::randu(source,cv::Scalar(0),cv::Scalar(255));
        cv::randu(target,cv::Scalar(0),cv::Scalar(255));
        logger->info("src_size:(h,w)=({0},{1}), target_size:(h,w)=({2},{3})",
            source.rows,source.cols,target.rows,target.cols);
        // my NCC test
        printf("source image size w,h = (%d,%d) \n", source.cols, source.rows);
        printf("target image size w,h = (%d,%d) \n", target.cols, target.rows);

        //warm up
        mycv::NormalizedCrossCorrelation(source, target, result);

        start_time = std::chrono::steady_clock::now();;
        for (int n = 0; n < TIMES; n++)
        {
            mycv::NormalizedCrossCorrelation(source, target, result);
        }
        end_time = std::chrono::steady_clock::now();;
        myncc_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / TIMES;
        printf("my NCC run %d times,average use %f ms \n", TIMES, myncc_runtime);

        cv::minMaxLoc(result, &min_value, &max_value, &min_loc, &max_loc);
        printf("min_value=%f , min_loc(x,y)=(%d,%d), \t max_value=%f,max_loc(x,y)=(%d,%d)\n",
            min_value, min_loc.x, min_loc.y, max_value, max_loc.x, max_loc.y);

        logger->info("my NCC run {0} times,average use {1} ms \n", TIMES, myncc_runtime);
        logger->info("my NCC min_value = {0}, min_loc(x, y) = ({1}, {2}), \t max_value = {3}, max_loc(x, y) = ({4}, {5})\n",
            min_value, min_loc.x, min_loc.y, max_value, max_loc.x, max_loc.y);

        //warm up
        cv::matchTemplate(source, target, result, cv::TM_CCOEFF_NORMED);

        // opencv NCC test
        start_time = std::chrono::steady_clock::now();;
        for (int n = 0; n < TIMES; n++)
        {
            cv::matchTemplate(source, target, result, cv::TM_CCOEFF_NORMED);
        }
        end_time = std::chrono::steady_clock::now();;
        opencv_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / TIMES;
        printf("opencv NCC run %d times,average use %f ms \n", TIMES, opencv_runtime);
        cv::minMaxLoc(result, &min_value, &max_value, &min_loc, &max_loc);
        printf("min_value=%f , min_loc(x,y)=(%d,%d), \t max_value=%f,max_loc(x,y)=(%d,%d)\n",
            min_value, min_loc.x, min_loc.y, max_value, max_loc.x, max_loc.y);


        logger->info("opencv NCC run {0} times,average use {1} ms \n", TIMES, opencv_runtime);
        logger->info("opencv NCC min_value = {0}, min_loc(x, y) = ({1}, {2}), \t max_value = {3}, max_loc(x, y) = ({4}, {5})\n",
            min_value, min_loc.x, min_loc.y, max_value, max_loc.x, max_loc.y);
        logger->info("speed : myncc_runtime / opencv_runtime = {}", (int)(myncc_runtime / opencv_runtime));
    }
  

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
    //cv::imwrite("data\\result.png", result);
}

void del()
{
    std::cout << DBL_MAX << std::endl;
    std::cout << FLT_MAX << std::endl;

}


int main()
{
    //test_NCC();
    //test_integralImage();
    //test_NCC_speed();
    cmp_speed();
    //del();
    system("pause");
    return 0;
}