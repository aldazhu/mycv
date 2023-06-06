#include "mycv.hpp"

#include <string>
#include <iostream>

#include "spdlog/logger.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "utils.h"

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
    cv::integral(source, opencv_sum,opencv_sqsum, CV_32F,CV_64F);
    mycv::showImage(source,"source");
    mycv::showImage(opencv_sum,"opencv integral");
    mycv::showImage(result, "integral",0);

    //another integral image
    cv::Mat integral_image, integral_sq;
    mycv::integralIPP(source, integral_image, integral_sq);

    
    
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
    //cv::imwrite("data\\integral.png", integral_image);
    //cv::imwrite("data\\integral_sq.png", integral_sq);


}

void test_NCC_speed()
{
    const int TIMES = 10;
    std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/source.jpg";
    std::string target_path = "H:/myProjects/work/mycv-master/mycv-master/data/target.jpg";
    std::string log_path = "ncc_speed.txt";
    cv::Mat source, target, result;
    source = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    target = cv::imread(target_path, cv::IMREAD_GRAYSCALE);
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
        //mycv::NormalizedCrossCorrelation(source, target, result);
        mycv::NormalizedCrossCorrelationFFT(source, target, result);

        start_time = std::chrono::steady_clock::now();;
        for (int n = 0; n < TIMES; n++)
        {
            //mycv::NormalizedCrossCorrelation(source, target, result);
            mycv::NormalizedCrossCorrelationFFT(source, target, result);
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
	std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/source.jpg";
	std::string target_path = "H:/myProjects/work/mycv-master/mycv-master/data/target.jpg";
    cv::Mat source, target,result;
    source = cv::imread(src_path,cv::IMREAD_GRAYSCALE);
    target = cv::imread(target_path, cv::IMREAD_GRAYSCALE);

    mycv::NormalizedCrossCorrelationFFT(source, target, result);
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


void Test_IntegralSpeed()
{
	mycv::Timer_ms t;
	constexpr int kTimes = 10;
	for (int src_size = 100; src_size < 2000; src_size += 200)
	{
		auto source = cv::Mat(cv::Size(src_size, src_size), CV_8UC1);
		cv::randu(source, cv::Scalar(0), cv::Scalar(255));
		cv::Mat integral, sq_integral;

		std::cout << "mycv integral run " << kTimes << " times ";
		t.Restart();
		for (int i = 0; i < kTimes; ++i)
			mycv::integral(source, integral, sq_integral);
		t.Duration();

		std::cout << "opencv integral run " << kTimes << " times ";
		t.Restart();
		for (int i = 0; i < kTimes; ++i)
			cv::integral(source, integral, sq_integral, CV_64FC1, CV_64FC1);
		t.Duration();

	}
}
	

#include <opencv2/opencv.hpp>
#include <immintrin.h> // 包含AVX指令集

using namespace cv;

void integral_avx(Mat& src, Mat& dst)
{
	int rows = src.rows, cols = src.cols;
	dst = Mat::zeros(rows + 1, cols + 1, CV_32FC1); // 初始化积分图
	if (src.type() == CV_8U)
	{
		src.convertTo(src, CV_32FC1);
	}

	// 按行计算积分图
	for (int i = 1; i <= rows; ++i)
	{
		float* sdata = src.ptr<float>(i - 1);
		float* idata = dst.ptr<float>(i) + 1;
		float* idata_prev = dst.ptr<float>(i - 1) + 1;
		__m256 row_sum = _mm256_setzero_ps(); // 初始化AVX向量

		// 按8个像素为一组进行计算
		for (int j = 0; j < cols; j += 8)
		{
			__m256 data = _mm256_loadu_ps((float*)(sdata + j)); // 加载8个像素值到AVX向量
			row_sum = _mm256_add_ps(row_sum, data); // 对8个像素值求和
			__m256 sum = _mm256_add_ps(row_sum, _mm256_loadu_ps((float*)(idata_prev + j))); // 前一行的累计和加上当前行的像素和
			_mm256_storeu_ps((float*)(idata + j), sum); // 将结果存储到积分图中
		}
	}
}

int Test_IntegralAVX()
{
	Mat src = imread("H:/myProjects/work/mycv-master/mycv-master/data/source.jpg", IMREAD_GRAYSCALE);
	if (src.empty())
		return -1;

	Mat integral_opencv, integral_avx2;
	double t1 = (double)getTickCount();
	integral(src, integral_opencv, CV_32F); // 使用OpenCV自带的积分图函数计算
	double t2 = (double)getTickCount();
	integral_avx(src, integral_avx2); // 使用AVX加速的积分图计算函数计算
	double t3 = (double)getTickCount();
	mycv::showImage( integral_opencv, "opencv integral");
	mycv::showImage( integral_avx2, "avx2 integral",0);
	std::cout << "OpenCV自带函数用时: " << (t2 - t1) / getTickFrequency() << " s" << std::endl;
	std::cout << "AVX加速用时: " << (t3 - t2) / getTickFrequency() << " s" << std::endl;

	return 0;
}

int main()
{
	Test_IntegralAVX();
    //test_NCC();
    //test_integralImage();
	//Test_IntegralSpeed();
    //test_NCC_speed();
    //cmp_speed();
    //del();
    system("pause");
    return 0;
}