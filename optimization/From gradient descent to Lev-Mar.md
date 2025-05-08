# 牛顿法
## 牛顿法求解方程的数学原理
解方程$f(x)=0$，当前迭代值为$x_n$，在$x_n$处泰勒展开取一阶近似有$f(x)=f(x_n)+f'(x_n)(x-x_n)$，令一阶近似为0，有
$$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$$
基于此迭代公式逼近$f(x)=0$的根。

每一次更新使用曲线在当前近似解$x_n$处的切线代替原来的曲线，将切线与x轴的交点作为新的近似解。

## 收敛性与条件
牛顿法在单根附近具有平方收敛速度（即误差随迭代次数呈平方级减少），前提是初始值足够接近真实解且函数在邻域内二阶可导。对于重根的情况，收敛速度会退化为线性。

## 多元向量值函数的情况
解方程$F(\mathbf x)=0$，当前迭代向量为$\mathbf x^{(k)}$，泰勒展开有
$$F(\mathbf x)=F(\mathbf x^{(k)})+J(\mathbf x^{(k)})(\mathbf x-\mathbf x^{(k)})$$
其中

$$J(\mathbf x)=\begin{bmatrix}\frac{\partial F_1}{\partial x_1}&\dotsm&\frac{\partial F_1}
{\partial x_n}\\\dotsm&\ddots&\dotsm\\\frac{\partial F_m}{\partial x_1}&\dotsm&\frac{\partial F_m}
{\partial x_n}\end{bmatrix}$$

迭代公式为
$$x^{(k+1)}=x^{(k)}-J^{-1}(x^{(k)})\cdot F(x^{(k)})$$

若$F$是多元标量函数，即$m=1$，则雅可比矩阵为$1\times n$的向量。若$n=1$则退化为一元函数的情况，否则需要求伪逆。在实现中通常不会显示求雅可比矩阵的伪逆，而是直接求解方程组$J(\mathbf x^{(k)})(\mathbf x-\mathbf x^{(k)})=-F(\mathbf x^{(k)})$。


## 牛顿法求解优化问题
求解优化问题（求极值）可转化为求函数的一阶导函数的根，使用牛顿法解方程的方法，可以得到迭代公式为
$$x_{n+1}=x_n-\frac{f^{'}(x_n)}{f^{''}(x_n)}$$

扩展到多元标量函数$f(\mathbf x)$的场合，转化为向量值函数$J_f(\mathbf x)$（或者等价地，梯度$\nabla f$）的求零点问题。
$$\mathbf x^{(k+1)}=\mathbf x^{(k)}-H^{-1}(\mathbf x^{(k)})\nabla f(\mathbf x^{(k)})$$
$H_f(\mathbf x)$为$f$的海森矩阵，即$\nabla f$的雅可比矩阵。
由此可以看出牛顿法求解极值不仅需要求函数的海森矩阵，还需要求海森矩阵的逆。

# 拟牛顿法
## 替代策略
构造$H_k\approx H^{-1}(\mathbf x_k)$或$B_k\approx H(\mathbf x_k)$并满足以下拟牛顿条件/割线方程（secant condition）
$$B_{k+1}(\mathbf x_{k+1}-\mathbf x_{k})=\nabla f(\mathbf x_{k+1})-\nabla f(\mathbf x_{k})$$
或者等价地
$$H_{k+1}(\nabla f(\mathbf x_{k+1})-\nabla f(\mathbf x_{k}))=\mathbf x_{k+1}-\mathbf x_{k}$$

这个近似的思想也非常的直观：海森矩阵原意为梯度的雅可比矩阵，相当于梯度变化量的一阶近似，现在直接替换为梯度变化量本身。

## BFGS
以构造$B_k$为例的BFGS算法，割线方程为
$$\begin{aligned}B_{k+1}(\mathbf x_{k+1}-\mathbf x_{k})&=\nabla f(\mathbf x_{k+1})-\nabla f(\mathbf x_{k})\\B_{k+1}\mathbf s_{k}&=\mathbf y_{k}\end{aligned}$$

除了需要满足割线方程外，我们还希望$B_{k+1}$是$B_{k}$的对称修正，并且尽可能接近$B_{k}$，因此更新$B_{k+1}$转化为带约束的优化问题
$$\min_{B_{k+1}}||B_{k+1}-B_{k}||_F$$
约束条件为$B_{k+1}\mathbf s_{k}=\mathbf y_{k}$且$B_{k+1}$对称。

秩二修正更新
$$B_{k+1}=B_{k}-\frac {B_k\mathbf s_k\mathbf s_k^TB_{k}}{\mathbf s_k^TB_k\mathbf s_k}+\frac{\mathbf y_k\mathbf y_k^T}{\mathbf y_k^T\mathbf s_k}$$

# 高斯牛顿法
## 非线性最小二乘优化问题
$$\min_{\mathbf x}F(\mathbf x)=\frac{1}{2}\sum_{i=1}^{m}[r_i(\mathbf x)]^2$$

其中$r_i(\mathbf x)$为第$i$个残差函数。
对残差函数（注意，跟面向一般优化问题的牛顿法起手不同之处，这里不是对优化函数进行泰勒展开，而是对残差函数）进行泰勒展开与线性化，有

$$r_i(\mathbf x_k+\Delta \mathbf x)=r_i(\mathbf x_k)+J_i(\mathbf x_k)\Delta \mathbf x$$

则优化目标函数可近似为

$$\begin{aligned}F(\mathbf x)&=\frac{1}{2}\sum_{i=1}^{m}[r_i(\mathbf x)]^2\\&\approx \frac{1}{2}[r_i(\mathbf x_k)+J_i(\mathbf x_k)\Delta \mathbf x]^2\\&=\frac{1}{2}(\mathbf r(\mathbf x_k)^T\mathbf r(\mathbf x_k)+2\mathbf r(\mathbf x_k)^T\mathbf J(\mathbf x_k)\Delta \mathbf x+\Delta \mathbf x^T\mathbf J(\mathbf x_k)^T\mathbf J(\mathbf x_k)\Delta \mathbf x)\end{aligned}$$


其中$\mathbf J$是各个残差函数的雅可比矩阵的堆叠矩阵，即
$$J(\mathbf x)=\begin{bmatrix}\frac{\partial F_1}{\partial x_1}&\dotsm&\frac{\partial F_1}
{\partial x_n}\\\dotsm&\ddots&\dotsm\\\frac{\partial F_m}{\partial x_1}&\dotsm&\frac{\partial F_m}
{\partial x_n}\end{bmatrix}$$

求极值转为求解一阶导函数的零点，因此上述近似函数对$\Delta \mathbf x$求导并设为0（向量），有线性方程

$$\mathbf J(\mathbf x_k)^T\mathbf J(\mathbf x_k)\Delta \mathbf x=-\mathbf J(\mathbf x_k)^T\mathbf r(\mathbf x_k)$$

$$\Delta \mathbf x=-(\mathbf J(\mathbf x_k)^T\mathbf J(\mathbf x_k))^{-1}\mathbf J(\mathbf x_k)^T\mathbf r(\mathbf x_k)$$

注意，这里的$\mathbf J$是残差函数（而非优化目标函数）的雅可比矩阵。

## 与牛顿法的关联
本质上依然是牛顿法的思路，但由于优化问题是关于优化参数的非线性最小二乘函数，从而使用$J^TJ$替换牛顿法中的$H$。可以理解为非线性最小二乘优化问题下的拟牛顿法（使用某些量作为海森矩阵的估算）。

## 阻尼高斯-牛顿方程
防止$J^TJ$近似奇异导致数值不稳定，将线性方程改写为带阻尼项的形式，改善矩阵条件数。

$$（\mathbf J(\mathbf x_k)^T\mathbf J(\mathbf x_k)+\lambda \mathbf I）\Delta \mathbf x=-\mathbf J(\mathbf x_k)^T\mathbf r(\mathbf x_k)$$

# Levenberg-Marquardt
## 数学原理
同样是面向非线性最小二乘优化问题。在阻尼高斯-牛顿方程的基础上，增加阻尼因子的自适应调整。

$$（\mathbf J(\mathbf x_k)^T\mathbf J(\mathbf x_k)+\lambda \mathbf I）\Delta \mathbf x=-\mathbf J(\mathbf x_k)^T\mathbf r(\mathbf x_k)$$

## 阻尼因子自适应调整策略
阻尼因子$\lambda$的调整原则：目标函数值减小，则接受更新并减小$\lambda$；否则增大$\lambda$并重新计算
注意，$\lambda=0$相当于高斯牛顿法，$\lambda\gg1$相当于（步长很小的）梯度下降法。

