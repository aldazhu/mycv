#include "mycv.hpp"

#include <string>
#include <iostream>

#include "spdlog/logger.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "mycv.hpp"

#include "threshold.h"

#include "utils.h"

void test_CalMeanVarSpeed()
{
    const int TIMES = 1000;

    std::chrono::steady_clock::time_point start_time, end_time;
    double abs_runtime = 0, integral_runtime = 0;
    mycv::Timer_ms ts;
    for (int size = 100; size < 1000; size += 100)
    {
        cv::Mat target = cv::Mat(cv::Size(size, size), CV_8UC1);
        cv::randu(target, cv::Scalar(0), cv::Scalar(255));
        spdlog::info("\n \n image size(h, w) = ({}, {})",target.rows,target.cols);
        int t_h = target.rows;
        int t_w = target.cols;
        const double target_size = (double)t_h * t_w;
        double target_region_sum, target_region_sqsum, target_mean, target_var, target_std_var;


        cv::Mat mean_mat, stddev_mat;
        spdlog::info("opencv 直接计算target的均值和方差");
        ts.Restart();
        for (int times = 0; times < TIMES; times++)
        {
            cv::meanStdDev(target, mean_mat, stddev_mat);
        }
        abs_runtime = ts.Duration();
        spdlog::info("run {0} times, use {1}ms", TIMES, abs_runtime);
        spdlog::info("opencv mean:{}, std variance:{}\n\n", mean_mat.at<double>(0), stddev_mat.at<double>(0));

        spdlog::info("mycv 直接计算target的均值和方差");
        ts.Restart();
        for (int times = 0; times < TIMES; times++)
        {
            target_mean = mycv::calculateMean(target);
            target_var = mycv::calculateVariance(target, target_mean);
            target_std_var = std::sqrt(target_var);
 
        }
        abs_runtime = ts.Duration();
        spdlog::info("run {0} times, use {1}ms", TIMES, abs_runtime);
        spdlog::info("mycv mean:{}, std variance:{}\n\n", target_mean, target_std_var);

        spdlog::info("mycv 递推计算均值和方差");
        float mycv_mean, mycv_var, mycv_stdvar;
        ts.Restart();
        for (int times = 0; times < TIMES; times++)
        {
            mycv::CalMeanVar(target, mycv_mean, mycv_var);
            mycv_stdvar = std::sqrt(mycv_var);
        }
        abs_runtime = ts.Duration();
        spdlog::info("run {0} times, use {1}ms", TIMES, abs_runtime);
        spdlog::info("mycv mean:{}, std variance:{}", mycv_mean, mycv_stdvar);
       

        cv::Mat target_sum, target_sqsum;
        spdlog::info("积分图的方法计算target的均值和方差");
        ts.Restart();
        for (int times = 0; times < TIMES; times++)
        {
            mycv::integral(target, target_sum, target_sqsum);
            //cv::integral(target, target_sum, target_sqsum,CV_64F,CV_64F);
            
            mycv::getRegionSumFromIntegralImage(target_sum, 0, 0, target.cols - 1, target.rows - 1, target_region_sum);
            mycv::getRegionSumFromIntegralImage(target_sqsum, 0, 0, target.cols - 1, target.rows - 1, target_region_sqsum);
            target_mean = target_region_sum / target_size;
            target_var = (target_region_sqsum - target_mean * target_region_sum) / target_size;
            target_std_var = std::sqrt(target_var);
        }
        abs_runtime = ts.Duration();
        spdlog::info("run {0} times,  use {1}ms", TIMES, abs_runtime);
        spdlog::info("mycv integral mean:{}, std variance:{}\n\n", target_mean, target_std_var);
        

    }
    
}

void test_error_code()
{
    int code = mycv::kImageEmpty;
    std::string msg = "test error";
    MYCV_ERROR2(code, msg);

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
    constexpr int pyramid_level = 5;
    std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/source.jpg";
    std::string target_path = "H:/myProjects/work/mycv-master/mycv-master/data/target2.jpg";
    std::string log_path = "ncc_speed.txt";
    cv::Mat source, target, result,src,tar;
    src = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    tar = cv::imread(target_path, cv::IMREAD_GRAYSCALE);
    source = src;
    target = tar;
    std::chrono::steady_clock::time_point start_time,end_time;
    double myncc_runtime = 0, opencv_runtime = 0;

    float x, y, score;

    auto logger = spdlog::basic_logger_mt("NCC", log_path);
    logger->set_level(spdlog::level::err);
    // location
    double min_value, max_value;
    cv::Point min_loc, max_loc;
    for (int src_size = 500; src_size <= 1200; src_size += 100)
    {
        //float ratio = (float)src_size / (float)src.cols;
        //cv::resize(src, source, cv::Size(), ratio, ratio);
        //cv::resize(tar, target, cv::Size(), ratio, ratio);
        
        logger->info("src_size:(h,w)=({0},{1}), target_size:(h,w)=({2},{3})",
            source.rows,source.cols,target.rows,target.cols);
        // my NCC test
        printf("source image size w,h = (%d,%d) \n", source.cols, source.rows);
        printf("target image size w,h = (%d,%d) \n", target.cols, target.rows);

        //warm up
        //mycv::NormalizedCrossCorrelation(source, target, result);
        //mycv::NormalizedCrossCorrelationFFT(source, target, result);

        start_time = std::chrono::steady_clock::now();;
        for (int n = 0; n < TIMES; n++)
        {
            mycv::NCCPyramid(source, target, pyramid_level, x, y, score);
            //mycv::NormalizedCrossCorrelation(source, target, result);
            //mycv::FastNormalizedCrossCorrelation(source, target, result);
            //mycv::NormalizedCrossCorrelationFFT(source, target, result);
        }
        end_time = std::chrono::steady_clock::now();
        myncc_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / TIMES;
        printf("my NCC run %d times,average use %f ms \n", TIMES, myncc_runtime);

        //cv::minMaxLoc(result, &min_value, &max_value, &min_loc, &max_loc);
        //printf("min_value=%f , min_loc(x,y)=(%d,%d), \t max_value=%f,max_loc(x,y)=(%d,%d)\n",
        //    min_value, min_loc.x, min_loc.y, max_value, max_loc.x, max_loc.y);
        printf("min_value=%f , min_loc(x,y)=(%d,%d), \t max_value=%f,max_loc(x,y)=(%d,%d)\n",
            0.0, 0, 0, score, (int)x, (int)y);

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

        printf("opencv run faster %f times\n", myncc_runtime / opencv_runtime);
    }
  

}

void test_NCC()
{
	std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/source.jpg";
	std::string target_path = "H:/myProjects/work/mycv-master/mycv-master/data/target2.jpg";
    cv::Mat source, target,result;
    source = cv::imread(src_path,cv::IMREAD_GRAYSCALE);
    target = cv::imread(target_path, cv::IMREAD_GRAYSCALE);

    mycv::FastNormalizedCrossCorrelation(source, target, result);
    mycv::showImage(target, "target");
    mycv::showImage(source, "source");
    mycv::showImage(result, "result", 0);

    result += 1;
    result *= 128;
    result.convertTo(result, CV_8U);
    //cv::imwrite("data\\result.png", result);
}

void test_OTSU()
{
    std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/source.jpg";
   
    cv::Mat source, result,opencv_result;
    source = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
    int opencv_th = cv::threshold(source, opencv_result,0,255, cv::THRESH_OTSU | cv::THRESH_BINARY);
    std::cout << "opencv thresh : " << opencv_th << std::endl;
    mycv::showImage(opencv_result, "opencv result");

    int th = mycv::OTSU(source, result,0);
    std::cout << "otsu thresh : " << th << std::endl;
    mycv::showImage(source, "source", 1);
    mycv::showImage(result, "result", 0);
    mycv::OTSU(source, result, 1);
    mycv::showImage(result, "result", 0);
    mycv::OTSU(source, result, 2);
    mycv::showImage(result, "result", 0);
    mycv::OTSU(source, result, -1);
    mycv::showImage(result, "result", 0);


}

void test_OTSU_speed()
{
    std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/target.jpg";

    cv::Mat source, result, opencv_result;
    source = cv::imread(src_path, cv::IMREAD_GRAYSCALE);

    constexpr int kTimes = 1000;
    std::cout << "opencv OTSU:";

    mycv::Timer_us t;
    for (int i = 0; i < kTimes; ++i)
        cv::threshold(source, opencv_result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    t.Duration();

    std::cout << "mycv OTSU:";
    t.Restart();
    for (int i = 0; i < kTimes; ++i)
        mycv::OTSU(source, result,0);
    t.Duration();


}

void test_hist()
{
    std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/target.jpg";

    cv::Mat source, result, opencv_result;
    source = cv::imread(src_path);

    int hist[256] = { 0 };
    int hist_avx[256] = { 0 };
    mycv::GetHist(source, hist);
    mycv::GetHistAVX(source, hist_avx);

    for (int i = 0; i < 256; ++i)
    {
        std::cout << hist[i] << "," << hist_avx[i] << std::endl;
    }


}



void del()
{
    std::cout << DBL_MAX << std::endl;
    std::cout << FLT_MAX << std::endl;

}


void test_IntegralSpeed()
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
	
void test_Pyramid()
{
    std::string src_path = "H:/myProjects/work/mycv-master/mycv-master/data/source.jpg";
   
    cv::Mat source = cv::imread(src_path);

    std::vector<cv::Mat> py_images;
    int level = 3;
    mycv::BuildPyramidImages(source, py_images, level);
    for (int i=0; i<py_images.size();++i)
    {
        auto image = py_images[i];
        mycv::showImage(image, std::to_string(i), 1, 1);
    }
    cv::waitKey(0);

}


float vectorDotProductAVX(const float* vectorA, const float* vectorB, int length) {
    int avxLength = length / 8; // 每次处理 8 个浮点数

    __m256 sum = _mm256_setzero_ps(); // 初始化累加和为 0

    for (int i = 0; i < avxLength; ++i) {
        __m256 vecA = _mm256_loadu_ps(vectorA + i * 8); // 加载 8 个浮点数到 AVX 寄存器
        __m256 vecB = _mm256_loadu_ps(vectorB + i * 8); // 加载 8 个浮点数到 AVX 寄存器

        sum = _mm256_add_ps(sum, _mm256_mul_ps(vecA, vecB)); // 使用 AVX 指令进行乘法和累加
    }

    float dotProduct = 0.0f;
    // 处理剩余的元素
    for (int i = avxLength * 8; i < length; ++i) {
        dotProduct += vectorA[i] * vectorB[i];
    }

    // 将结果从 AVX 寄存器取回
    float result[8];
    _mm256_storeu_ps(result, sum);

    // 对结果求和，得到最终的内积
    
    for (int i = 0; i < 8; ++i) {
        dotProduct += result[i];
    }

    return dotProduct;
}
float vectorDotProduct(const float* vectorA, const float* vectorB, int length)
{
    float dot_product = 0.0f;
    for (size_t i = 0; i < length; i++)
    {
        dot_product += vectorA[i] * vectorB[i];
    }
    return dot_product;
}

void test_vectorDotProductAVX()
{
    for (int length = 100; length < 10000; length += 100)
    {
        std::cout << "vector length :" << length << std::endl;
        constexpr int times = 100000;
        std::unique_ptr<float[]> a = std::make_unique<float[]>(length);
        std::unique_ptr<float[]> b = std::make_unique<float[]>(length);
        for (int i = 0; i < length; ++i)
        {
            a[i] = rand() % 10;
            b[i] = rand() % 10;
        }

        mycv::Timer_us ts;
        float result = 0;

        std::cout << "AVX: ";
        ts.Restart();
        for (int i = 0; i < times; ++i)
            result = vectorDotProductAVX(a.get(), b.get(), length);
        ts.Duration();
        std::cout << "res: " << result << std::endl;

        std::cout << "notmal:";
        ts.Restart();
        for (int i = 0; i < times; ++i)
            result = vectorDotProduct(a.get(), b.get(), length);
        ts.Duration();
        std::cout << "res: " << result << std::endl;;
    }

}


void test_calculateCovarianceAVX()
{
    const int times = 1000;
    
    for (int image_size = 10; image_size < 500; image_size += 10) {
        cv::Mat target = cv::Mat(cv::Size(image_size, image_size), CV_8UC1);
        cv::Mat target2 = cv::Mat(cv::Size(image_size, image_size), CV_8UC1);
        cv::randu(target, cv::Scalar(0), cv::Scalar(255));
        cv::randu(target2, cv::Scalar(0), cv::Scalar(255));

        std::cout << "image size: " << image_size << std::endl;

        mycv::Timer_ms ts;
        double mean1, mean2;
        mean1 = mycv::calculateMean(target);
        mean2 = mycv::calculateMean(target2);

        double conv = 0;
        std::cout << "normal calculateCovariance";
        ts.Restart();
        for (size_t i = 0; i < times; i++)
        {
            conv = mycv::calculateCovariance(target, target2, mean1, mean2);
        }
        ts.Duration();
        std::cout << "conv: " << conv << std::endl;

        std::cout << "avx calculateCovarianceAVX";
        ts.Restart();
        for (size_t i = 0; i < times; i++)
        {
            conv = mycv::calculateCovarianceAVX(target, target2, mean1, mean2);
        }
        ts.Duration();
        std::cout << "conv: " << conv << std::endl;

        std::cout << "avx calculateCovarianceAVXFlatten";
        ts.Restart();
        for (size_t i = 0; i < times; i++)
        {
            conv = mycv::calculateCovarianceAVXFlatten(target, target2, mean1, mean2);
        }
        ts.Duration();
        std::cout << "conv: " << conv << std::endl;
    }
}

void del_avx()
{
    cv::Mat A = cv::Mat::ones(cv::Size(200, 200), CV_32FC1)*255;
    

}

int main()
{

    //test_OTSU();
    //test_OTSU_speed();
    //test_hist();

	//test_IntegralAVX();
    //test_NCC();
    //test_integralImage();
	//test_IntegralSpeed();
    //test_NCC_speed();
    //test_CalMeanVarSpeed();
    //del();
    
    //test_Pyramid();
    //test_vectorDotProductAVX();
   
    test_calculateCovarianceAVX();

    //system("pause");
    return 0;
}