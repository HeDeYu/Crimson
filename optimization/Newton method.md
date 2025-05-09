# 牛顿法解方程
## 数学原理
解方程$f(x)=0$，当前迭代值为$x_n$，在$x_n$处泰勒展开取一阶近似有$f(x)=f(x_n)+f'(x_n)(x-x_n)$，令一阶近似为0，有
$$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$$
基于此迭代公式逼近$f(x)=0$的根。

每一次更新使用曲线在当前近似解$x_n$处的切线代替原来的曲线，将切线与x轴的交点作为新的近似解。

## 收敛性与条件
牛顿法在单根附近具有平方收敛速度（即误差随迭代次数呈平方级减少），前提是初始值足够接近真实解且函数在邻域内二阶可导。对于重根的情况，收敛速度会退化为线性。

## 多元函数的情况
解方程$F(x)=[f_1(x),...,f_n(x)]^T$，当前迭代值为$x^{(k)}$，泰勒展开有
$$F(x)=F(x^{(k)})+J(x^{(k)})$$

# 牛顿法求解优化问题