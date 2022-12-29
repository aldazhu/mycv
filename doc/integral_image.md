# <center> 积分图 integral image

## 1. 原理：

Summed Area Table是一种数据结构和算法，用于快速有效地生成网格矩形子集中的值总和。在图像处理领域，它也被称为积分图像。

记源图像为$I_{m \times n}$，积分图为SAT（Summed Area Table）则
$$SAT(x,y) = \underset{i\leq x,j\leq y}{\Sigma }I(i,j)$$

$$SQAT(x,y) = \underset{i\leq x,j\leq y}{\Sigma }I(i,j)^2$$

$$SAT(x,y) = SAT(x-1,y) + SAT(x,y-1) - SAT(x-1,y-1) + I(x,y)$$ 

## 2. 示例

![example](../data/Integral_image_example.png "example" )
图片摘自Wikipedia。1为源图，2为积分图每一个像素点的值都是它在源图上对应位置的上面、左边所有像素值的和。

## 3. 计算区域均值

当要计算源图中的某个矩形区域的像素和时可以用查表的方式快速计算，构建好积分图后求区域和的复杂度将是$O(1)$。

设在源图中的矩形$R$左上角坐标为$(tpx,tpy)$，左下角坐标为$(btx,bty)$,则计算该矩形区域内所有像素值和可以写为下式。

- 区域像素值之和
$$sum = SAT(btx,bty) - SAT(tpx,bty) - SAT(btx,tpy) + SAT(tpx,tpy)$$
- 区域均值
$$mean = \frac{sum}{(bty-tpy+1)(btx-tpx+1)}$$

## 4. 计算区域方差
计算区域方差
$$\sigma(S_{x,y})=\sqrt{var(S_{x,y})}=\sqrt{\frac{\Sigma_{i=1}^{m}\Sigma_{j=1}^{n}{(S_{x,y}(i,j)-\bar{S_{x,y}}})^2}{mn}}$$
let $A =  \Sigma_{i=1}^{m}\Sigma_{j=1}^{n}{(S_{x,y}(i,j)-\bar{S_{x,y}}})^2$ 则有
$$
\begin{aligned}
A &=\Sigma_{i=1}^{m}\Sigma_{j=1}^{n}( S_{x,y}(i,j)^2 -2\bar{S_{x,y}}S_{x,y}(i,j) + \bar{S_{x,y}}^2) \\
&= \underset{S_{x,y}}{SQAT(m,n)} -\underset{S_{x,y}}{2\bar{S_{x,y}}\times SAT(m,n)} + mn\times \bar{S_{x,y}}^2 \\
&=\underset{S_{x,y}}{SQAT(m,n)}-\underset{S_{x,y}}{2\bar{S_{x,y}}\times SAT(m,n)} + \bar{S_{x,y}}\times SAT(m,n) \\
&=\underset{S_{x,y}}{SQAT(m,n)}-\underset{S_{x,y}}{\bar{S_{x,y}}\times SAT(m,n)} 
\end{aligned}
$$


## 参考
[1] [https://en.wikipedia.org/wiki/Summed-area_table](https://en.wikipedia.org/wiki/Summed-area_table)
[2] [积分图(一) - 原理及应用 ](https://www.cnblogs.com/magic-428/p/9149868.html)