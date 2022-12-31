# <center> mycv

简介：
此项目用于自娱自乐！

# 实现
## 1. NCC
原理部分见 [template_match](doc\template_match.md)

## 2.积分图

### 速度测试
opencv中用TM_CCOEFF_NORMED方法
2022年12月18日
均值和方差都直接暴力计算

```
source image size w,h = (1095,680)
target image size w,h = (89,91)
my NCC run 10 times, use 12359.000000 ms       
min_value=-0.389405 , min_loc(x,y)=(892,537),    max_value=1.187444,max_loc(x,y)=(393,286)
opencv NCC run 10 times, use 14.000000 ms
min_value=-0.431418 , min_loc(x,y)=(799,210),    max_value=0.998322,max_loc(x,y)=(393,286)
```
opencv的速度是我的882.78倍。

采用积分图计算均值和方差
```
source image size w,h = (1095,680)
target image size w,h = (89,91)
my NCC run 10 times,average use 4741.000000 ms
min_value=-0.423899 , min_loc(x,y)=(892,537),    max_value=1.091333,max_loc(x,y)=(393,286)
opencv NCC run 10 times,average use 15.000000 ms
min_value=-0.431418 , min_loc(x,y)=(799,210),    max_value=0.998322,max_loc(x,y)=(393,286)
```
opencv的速度是我的316.06倍。