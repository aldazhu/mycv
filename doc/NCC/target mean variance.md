# <center> 计算模板图像的均值和方差
@[toc]
## 1. 分析：
在NCC中需要用到模板图像的均值和方差，在计算一个图像的均值和方差时有两种方式：
- 最直接的方式就是遍历像素直接计算，这样的话需要两次遍历图像，第一次遍历图像求得均值，第二次遍历求方差（计算方差要用到均值）；
- 另外一种是遍历一次图像建立两个积分图，用积分图来计算均值和方差

在我朴素的认知里，多次for循环的效率是不如把多条计算放到一个for循环里的，所以有一个初步的感觉是用积分图多用两个图的内存可以加快点速度，但是结果是相反的。

## 2. 速度对比验证

#### 2.1两种计算方式的对比代码
```cpp
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
            mycv::integral(target, target_sum, target_sqsum);
            //cv::integral(target, target_sum, target_sqsum,CV_64F,CV_64F);
            
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
            target_mean = mycv::calculateMean(target);
            target_var = mycv::calculateVariance(target, target_mean);
            target_std_var = std::sqrt(target_var);
            //cv::meanStdDev(target, mean_mat, stddev_mat);

        }
        end_time = std::chrono::steady_clock::now();

        abs_runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() ;
        spdlog::info("直接计算target的均值和方差");
        spdlog::info("run {0} times, use {1}ms", TIMES, abs_runtime);
        spdlog::info("mean:{}, std variance:{}", target_mean, target_std_var);
        //spdlog::info("opencv mean:{}, std variance:{}", mean_mat.at<double>(0), stddev_mat.at<double>(0));
        
        spdlog::info("abs_runtime / integral_image = {}",abs_runtime/integral_runtime);

    }
    
}
```
#### 2.2结果
```
[2022-12-31 19:07:49.802] [info]

 image size(h, w) = (100, 100)
[2022-12-31 19:07:49.842] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:49.843] [info] run 1000 times,  use 39ms
[2022-12-31 19:07:49.844] [info] mean:126.9468, std variance:73.48997462076035
[2022-12-31 19:07:49.863] [info] 直接计算target的均值和方差
[2022-12-31 19:07:49.863] [info] run 1000 times, use 18ms
[2022-12-31 19:07:49.863] [info] mean:126.9468, std variance:73.48997462076017
[2022-12-31 19:07:49.863] [info] abs_runtime / integral_image = 0.46153846153846156
[2022-12-31 19:07:49.864] [info]

 image size(h, w) = (200, 200)
[2022-12-31 19:07:50.014] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:50.015] [info] run 1000 times,  use 150ms
[2022-12-31 19:07:50.015] [info] mean:126.907375, std variance:73.66374851722777
[2022-12-31 19:07:50.086] [info] 直接计算target的均值和方差
[2022-12-31 19:07:50.087] [info] run 1000 times, use 69ms
[2022-12-31 19:07:50.087] [info] mean:126.907375, std variance:73.66374851722767
[2022-12-31 19:07:50.088] [info] abs_runtime / integral_image = 0.46
[2022-12-31 19:07:50.088] [info]

 image size(h, w) = (300, 300)
[2022-12-31 19:07:50.447] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:50.447] [info] run 1000 times,  use 358ms
[2022-12-31 19:07:50.447] [info] mean:127.00587777777778, std variance:73.75784567613849
[2022-12-31 19:07:50.604] [info] 直接计算target的均值和方差
[2022-12-31 19:07:50.604] [info] run 1000 times, use 156ms
[2022-12-31 19:07:50.604] [info] mean:127.00587777777778, std variance:73.75784567613874
[2022-12-31 19:07:50.605] [info] abs_runtime / integral_image = 0.43575418994413406
[2022-12-31 19:07:50.605] [info]

 image size(h, w) = (400, 400)
[2022-12-31 19:07:51.252] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:51.252] [info] run 1000 times,  use 645ms
[2022-12-31 19:07:51.253] [info] mean:126.97608125, std variance:73.51483648994535
[2022-12-31 19:07:51.535] [info] 直接计算target的均值和方差
[2022-12-31 19:07:51.536] [info] run 1000 times, use 281ms
[2022-12-31 19:07:51.537] [info] mean:126.97608125, std variance:73.51483648994528
[2022-12-31 19:07:51.538] [info] abs_runtime / integral_image = 0.4356589147286822
[2022-12-31 19:07:51.539] [info]

 image size(h, w) = (500, 500)
[2022-12-31 19:07:52.555] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:52.556] [info] run 1000 times,  use 1015ms
[2022-12-31 19:07:52.557] [info] mean:126.778828, std variance:73.62662175427047
[2022-12-31 19:07:52.993] [info] 直接计算target的均值和方差
[2022-12-31 19:07:52.993] [info] run 1000 times, use 434ms
[2022-12-31 19:07:52.994] [info] mean:126.778828, std variance:73.62662175427381
[2022-12-31 19:07:52.995] [info] abs_runtime / integral_image = 0.42758620689655175
[2022-12-31 19:07:52.996] [info]

 image size(h, w) = (600, 600)
[2022-12-31 19:07:54.473] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:54.473] [info] run 1000 times,  use 1475ms
[2022-12-31 19:07:54.474] [info] mean:127.02045, std variance:73.57434436312813
[2022-12-31 19:07:55.100] [info] 直接计算target的均值和方差
[2022-12-31 19:07:55.101] [info] run 1000 times, use 624ms
[2022-12-31 19:07:55.102] [info] mean:127.02045, std variance:73.57434436312465
[2022-12-31 19:07:55.103] [info] abs_runtime / integral_image = 0.4230508474576271
[2022-12-31 19:07:55.105] [info]

 image size(h, w) = (700, 700)
[2022-12-31 19:07:57.108] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:07:57.109] [info] run 1000 times,  use 2003ms
[2022-12-31 19:07:57.110] [info] mean:127.11293673469387, std variance:73.60538297187792
[2022-12-31 19:07:57.959] [info] 直接计算target的均值和方差
[2022-12-31 19:07:57.960] [info] run 1000 times, use 847ms
[2022-12-31 19:07:57.961] [info] mean:127.11293673469387, std variance:73.6053829719088
[2022-12-31 19:07:57.962] [info] abs_runtime / integral_image = 0.42286570144782826
[2022-12-31 19:07:57.964] [info]

 image size(h, w) = (800, 800)
[2022-12-31 19:08:00.614] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:08:00.615] [info] run 1000 times,  use 2649ms
[2022-12-31 19:08:00.616] [info] mean:126.9977671875, std variance:73.60779138669389
[2022-12-31 19:08:01.725] [info] 直接计算target的均值和方差
[2022-12-31 19:08:01.726] [info] run 1000 times, use 1108ms
[2022-12-31 19:08:01.727] [info] mean:126.9977671875, std variance:73.60779138669403
[2022-12-31 19:08:01.727] [info] abs_runtime / integral_image = 0.4182710456776142
[2022-12-31 19:08:01.730] [info]

 image size(h, w) = (900, 900)
[2022-12-31 19:08:05.110] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:08:05.111] [info] run 1000 times,  use 3379ms
[2022-12-31 19:08:05.112] [info] mean:126.94838024691359, std variance:73.61619958659867
[2022-12-31 19:08:06.535] [info] 直接计算target的均值和方差
[2022-12-31 19:08:06.536] [info] run 1000 times, use 1421ms
[2022-12-31 19:08:06.536] [info] mean:126.94838024691359, std variance:73.61619958660152
[2022-12-31 19:08:06.537] [info] abs_runtime / integral_image = 0.42053862089375554
请按任意键继续. . .
```
#### 2.3opencv的结果
积分图方法计算均值方差时用opencv的cv::integral方法，直接计算均值方差用cv::meanStdDev方法
```
[2022-12-31 19:12:36.645] [info]

 image size(h, w) = (100, 100)
[2022-12-31 19:12:36.660] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:36.660] [info] run 1000 times,  use 12ms
[2022-12-31 19:12:36.660] [info] mean:126.9468, std variance:73.48997462076035
[2022-12-31 19:12:36.661] [info] 直接计算target的均值和方差
[2022-12-31 19:12:36.661] [info] run 1000 times, use 0ms
[2022-12-31 19:12:36.661] [info] opencv mean:126.94680000000001, std variance:73.48997462076034
[2022-12-31 19:12:36.661] [info] abs_runtime / integral_image = 0
[2022-12-31 19:12:36.661] [info]

 image size(h, w) = (200, 200)
[2022-12-31 19:12:36.699] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:36.700] [info] run 1000 times,  use 37ms
[2022-12-31 19:12:36.700] [info] mean:126.907375, std variance:73.66374851722777
[2022-12-31 19:12:36.702] [info] 直接计算target的均值和方差
[2022-12-31 19:12:36.703] [info] run 1000 times, use 1ms
[2022-12-31 19:12:36.703] [info] opencv mean:126.907375, std variance:73.66374851722777
[2022-12-31 19:12:36.703] [info] abs_runtime / integral_image = 0.02702702702702703
[2022-12-31 19:12:36.703] [info]

 image size(h, w) = (300, 300)
[2022-12-31 19:12:36.784] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:36.785] [info] run 1000 times,  use 80ms
[2022-12-31 19:12:36.786] [info] mean:127.00587777777778, std variance:73.75784567613849
[2022-12-31 19:12:36.790] [info] 直接计算target的均值和方差
[2022-12-31 19:12:36.790] [info] run 1000 times, use 3ms
[2022-12-31 19:12:36.791] [info] opencv mean:127.00587777777778, std variance:73.75784567613849
[2022-12-31 19:12:36.791] [info] abs_runtime / integral_image = 0.0375
[2022-12-31 19:12:36.792] [info]

 image size(h, w) = (400, 400)
[2022-12-31 19:12:36.932] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:36.933] [info] run 1000 times,  use 140ms
[2022-12-31 19:12:36.934] [info] mean:126.97608125, std variance:73.51483648994535
[2022-12-31 19:12:36.941] [info] 直接计算target的均值和方差
[2022-12-31 19:12:36.942] [info] run 1000 times, use 6ms
[2022-12-31 19:12:36.942] [info] opencv mean:126.97608125000001, std variance:73.51483648994532
[2022-12-31 19:12:36.943] [info] abs_runtime / integral_image = 0.04285714285714286
[2022-12-31 19:12:36.944] [info]

 image size(h, w) = (500, 500)
[2022-12-31 19:12:37.169] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:37.170] [info] run 1000 times,  use 225ms
[2022-12-31 19:12:37.171] [info] mean:126.778828, std variance:73.62662175427047
[2022-12-31 19:12:37.181] [info] 直接计算target的均值和方差
[2022-12-31 19:12:37.182] [info] run 1000 times, use 9ms
[2022-12-31 19:12:37.183] [info] opencv mean:126.77882799999999, std variance:73.62662175427049
[2022-12-31 19:12:37.183] [info] abs_runtime / integral_image = 0.04
[2022-12-31 19:12:37.185] [info]

 image size(h, w) = (600, 600)
[2022-12-31 19:12:37.505] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:37.506] [info] run 1000 times,  use 320ms
[2022-12-31 19:12:37.507] [info] mean:127.02045, std variance:73.57434436312813
[2022-12-31 19:12:37.521] [info] 直接计算target的均值和方差
[2022-12-31 19:12:37.521] [info] run 1000 times, use 13ms
[2022-12-31 19:12:37.522] [info] opencv mean:127.02045000000001, std variance:73.57434436312812
[2022-12-31 19:12:37.523] [info] abs_runtime / integral_image = 0.040625
[2022-12-31 19:12:37.525] [info]

 image size(h, w) = (700, 700)
[2022-12-31 19:12:37.950] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:37.951] [info] run 1000 times,  use 424ms
[2022-12-31 19:12:37.952] [info] mean:127.11293673469387, std variance:73.60538297187792
[2022-12-31 19:12:37.970] [info] 直接计算target的均值和方差
[2022-12-31 19:12:37.971] [info] run 1000 times, use 17ms
[2022-12-31 19:12:37.972] [info] opencv mean:127.11293673469386, std variance:73.60538297187792
[2022-12-31 19:12:37.972] [info] abs_runtime / integral_image = 0.04009433962264151
[2022-12-31 19:12:37.974] [info]

 image size(h, w) = (800, 800)
[2022-12-31 19:12:38.541] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:38.542] [info] run 1000 times,  use 566ms
[2022-12-31 19:12:38.543] [info] mean:126.9977671875, std variance:73.60779138669389
[2022-12-31 19:12:38.568] [info] 直接计算target的均值和方差
[2022-12-31 19:12:38.568] [info] run 1000 times, use 23ms
[2022-12-31 19:12:38.569] [info] opencv mean:126.9977671875, std variance:73.60779138669392
[2022-12-31 19:12:38.570] [info] abs_runtime / integral_image = 0.04063604240282685
[2022-12-31 19:12:38.573] [info]

 image size(h, w) = (900, 900)
[2022-12-31 19:12:39.326] [info] 积分图的方法计算target的均值和方差
[2022-12-31 19:12:39.326] [info] run 1000 times,  use 752ms
[2022-12-31 19:12:39.327] [info] mean:126.94838024691359, std variance:73.61619958659867
[2022-12-31 19:12:39.357] [info] 直接计算target的均值和方差
[2022-12-31 19:12:39.358] [info] run 1000 times, use 29ms
[2022-12-31 19:12:39.360] [info] opencv mean:126.94838024691359, std variance:73.61619958659868
[2022-12-31 19:12:39.360] [info] abs_runtime / integral_image = 0.03856382978723404
请按任意键继续. . .
```

## 3. 结论
当要计算整个图像的均值和方差时用直接计算的方式要比积分图快得多！opencv4.5.4
以500*500的图像为例：
|版本|直接计算1000次用时(ms)|积分图计算1000次用时(ms)|倍数|
|-|-|-|-|
|mycv|434|1015|0.427|
|opencv|9|225|0.04|

## 4. 源码

### 4.1 直接计算均值和方差的代码

```cpp
/**
 * @brief 计算输入图像的方差，如果已知mean就不再计算mean
 * 
 * @param image  : 输入图CV_8UC1
 * @param mean  : 图像的灰度均值，默认值为-1，不输入时会计算mean
 * @return double ：图像的方差
 */
double calculateVariance(const cv::Mat &image,double mean)
{
    if (image.empty())  
    {
        MYCV_ERROR(kImageEmpty,"empty image");
        return -1;//正常的方差不会小于0
    }
    if (-1 == mean)
    {
        mean = calculateMean(image);
    }

    double sum = 0 ;
    for (int  row = 0; row < image.rows; row++)
    {
        const uchar * p = image.ptr<uchar>(row);
        for (int col = 0; col < image.cols; col++)
        {
            sum += (p[col] - mean) * (p[col] - mean);
        }
        
    }

    double var = sum / ((double)image.cols * (double)image.rows);
    
    return var;    
}



/**
 * @brief 计算输入图的灰度均值
 * 
 * @param image  : 输入图CV_8UC1
 * @return double ： 输入图像的灰度均值
 */
double calculateMean(const cv::Mat &image)
{
     if (image.empty())  
    {
        MYCV_ERROR(kImageEmpty,"empty image");
        return -1;
    }

    double sum = 0 ;
    for (int  row = 0; row < image.rows; row++)
    {
        const uchar * p = image.ptr<uchar>(row);
        for (int col = 0; col < image.cols; col++)
        {
            sum += p[col];
        }
        
    }

    double mean = sum / ((double)image.cols * (double)image.rows);
    return mean;
}
```

### 4.2 积分图计算均值和方差的代码
```cpp
/**
 * @brief 计算输入图的积分图,为了提高计算效率，可以让积分图比输入图多一行一列，
 * 具体的就是在原图左边插入一列0，上面插入一行0，设原图为I，积分图为SAT(summed area table)
 * 则：SAT(i,j) = SAT(i,j-1) + SAT(i-1,j) - SAT(i-1,j-1) + I(i,j)
 * SQAT(i,j) = SQAT(i,j-1) + SQAT(i-1,j) - SQAT(i-1,j-1) + I(i,j) * I(i,j)
 * 这样就不用考虑下边界的情况，省掉很多判断条件
 * 
 * @param image  : 输入图CV_8UC1，MxN
 * @param integral_image  : 积分图CV_32FC1,(M+1)x(N+1)
 * @param integral_sq : 平方的积分图CV_32FC1,(M+1)x(N+1)
 * @return int 
 */
int integral(const cv::Mat &image,cv::Mat &integral_image,cv::Mat &integral_sq)
{
     if(image.empty())
    {
        MYCV_ERROR(kImageEmpty,"image empty");
        return kImageEmpty;
    }

    int h = image.rows;
    int w = image.cols;
    integral_image = cv::Mat::zeros(cv::Size(w+1,h+1),CV_64FC1);
    integral_sq = cv::Mat::zeros(cv::Size(w+1,h+1),CV_64FC1);

    //SAT(i,j) = SAT(i,j-1) + SAT(i-1,j) - SAT(i-1,j-1) + I(i,j)
    //SQAT(i,j) = SQAT(i,j-1) + SQAT(i-1,j) - SQAT(i-1,j-1) + I(i,j) * I(i,j)
    for(int i = 0; i < h ; i++)
    {
        const uchar *ps = image.ptr<uchar>(i);
        double *pd_m1 = integral_image.ptr<double>(i);//integral 的"上一行"
        double *pd = integral_image.ptr<double>(i+1); //integral 的"当前行"
        double *pqd_m1 = integral_sq.ptr<double>(i);
        double *pqd = integral_sq.ptr<double>(i+1);
        for(int j = 0; j < w; j++)
        {
            pd[j+1] = pd[j] + pd_m1[j+1] - pd_m1[j] + (double)ps[j];
            pqd[j+1] = pqd[j] + pqd_m1[j+1] - pqd_m1[j] + (double)ps[j] * (double)ps[j];
        }
    }


    return kSuccess;
}


/**
 * @brief Get the Region sum From Integral Image or sq integral image
 * 原图上的区域为tpx，tpy,btx,bty,在积分图或者平方的积分图上的位置为tpx+1,tpy+1,btx+1,bty+1
 * region sum = SAT(btx+1,bty+1) - SAT(tpx,bty+1) - SAT(btx+1,tpy) + SAT(tpx,tpy)
 * 
 * @param integral  : 像素和的积分图或者像素平方的积分图，CV_64FC1格式
 * @param tpx  : x of top left
 * @param tpy  : y of top right
 * @param btx  : x of bottom right
 * @param bty  : y of bottom right
 * @param sum  : 区域和
 * @return int : 程序运行状态码
 */
int getRegionSumFromIntegralImage(const cv::Mat & integral,int tpx,int tpy,int btx,int bty,double &sum)
{
    if(integral.empty())
    {
        MYCV_ERROR(mycv::kImageEmpty,"Input image is empty!");
        return mycv::kImageEmpty;
    }
    if(tpx > btx 
    || tpy > bty
    || tpx < 0 
    || tpy < 0
    || btx > integral.cols - 1
    || bty > integral.rows - 1)
    {
        MYCV_ERROR(mycv::kBadSize,"0 <= tpx <= btx <= w, && 0<= tpy <= bty <= h");
        return mycv::kBadSize;
    }
    const double *ptp = integral.ptr<double>(tpy);
    const double *pbt = integral.ptr<double>(bty+1);
    
    sum = (*(pbt+btx+1)) - (*(pbt+tpx)) - (*(ptp+btx+1)) + (*(ptp+tpx)); 

    return mycv::kSuccess;
}
```