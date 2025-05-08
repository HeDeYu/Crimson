# 雅可比矩阵

$$J(\mathbf x)=\begin{bmatrix}\frac{\partial F_1}{\partial x_1}&\dotsm&\frac{\partial F_1}
{\partial x_n}\\\dotsm&\ddots&\dotsm\\\frac{\partial F_m}{\partial x_1}&\dotsm&\frac{\partial F_m}
{\partial x_n}\end{bmatrix}$$

## 雅可比矩阵与梯度的关联

# 雅可比矩阵求逆

## Moore-Penrose伪逆
对多元标量函数的雅可比矩阵$J_f \isin \R^{1\times n}$，其伪逆为
$$J_f^+=J_f^T(J_fJ_f^T)^{-1}$$