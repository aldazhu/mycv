# 积分图 integral image

定义：
SAT: sum area table
$$SAT(x,y) = \underset{i\leq x,j\leq y}{\Sigma }I(i,j)$$

$$SQAT(x,y) = \underset{i\leq x,j\leq y}{\Sigma }I(i,j)^2$$

$$SAT(x,y) = SAT(x-1,y) + SAT(x,y-1) - SAT(x-1,y-1) + I(x,y)$$ 

计算区域均值
tpx,tpy,btx,bty
AAAAA
ABAAA
AAABA
AAAAA


$area = SAT(btx,bty) - SAT(tpx,bty) - SAT(btx,tpy) + SAT(tpx,tpy)$


计算区域方差
$$\sigma(S_{x,y})=\sqrt{var(S_{x,y})}=\sqrt{\frac{\Sigma_{i=1}^{m}\Sigma_{j=1}^{n}{(S_{x,y}(i,j)-\bar{S_{x,y}}})^2}{mn}}$$
let $A =  \Sigma_{i=1}^{m}\Sigma_{j=1}^{n}{(S_{x,y}(i,j)-\bar{S_{x,y}}})^2$ 则有
$$
\begin{aligned}
A &=\Sigma_{i=1}^{m}\Sigma_{j=1}^{n}( S_{x,y}(i,j)^2 -2\bar{S_{x,y}}S_{x,y}(i,j) + \bar{S_{x,y}}^2) \\
&= \underset{S_{x,y}}{SQAT(m,n)} -\underset{S_{x,y}}{2\bar{S_{x,y}}\times SAT(m,n)} + mn\times \bar{S_{x,y}}^2
\end{aligned}
$$