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

# levmar库源码
注意源码中的m与n跟前述内容的含义对调了。
```
//C实现，用预编译宏来实现泛型编程（双精度与单精度）
int LEVMAR_DER(
  // m维优化向量p产生n维预测向量hx的多元向量值函数，与n维目标向量x求解最小二乘优化问题
  void (*func)(LM_REAL *p, LM_REAL *hx, int m, int n, void *adata),
  // m维优化向量p产生n维预测向量x~的多元向量值函数的雅可比矩阵mxn
  void (*jacf)(LM_REAL *p, LM_REAL *j, int m, int n, void *adata),
  LM_REAL *p,         // m维向量p，初始值及最终估算值
  LM_REAL *x,         // n维目标向量x
  int m,              
  int n,              
  int itmax,          // 最大迭代次数
  LM_REAL opts[4],    /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
                       * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
                       */
  LM_REAL info[LM_INFO_SZ],
					           /* O: information regarding the minimization. Set to NULL if don't care
                      * info[0]= ||e||_2 at initial p.
                      * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
                      * info[5]= # iterations,
                      * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                      *                                 2 - stopped by small Dp
                      *                                 3 - stopped by itmax
                      *                                 4 - singular matrix. Restart from current p with increased mu 
                      *                                 5 - no further error reduction is possible. Restart with increased mu
                      *                                 6 - stopped by small ||e||_2
                      *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
                      * info[7]= # function evaluations
                      * info[8]= # Jacobian evaluations
                      * info[9]= # linear systems solved, i.e. # attempts for reducing error
                      */
  LM_REAL *work,     /* working memory at least LM_DER_WORKSZ() reals large, allocated if NULL */
  LM_REAL *covar,    /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
  void *adata)       /* pointer to possibly additional data, passed uninterpreted to func & jacf.
                      * Set to NULL if not needed
                      */
{
register int i, j, k, l;
int worksz, freework=0, issolved;
/* temp work arrays */
LM_REAL *e,          /* nx1 */
       *hx,         /* \hat{x}_i, nx1 */
       *jacTe,      /* J^T e_i mx1 */
       *jac,        /* nxm */
       *jacTjac,    /* mxm */
       *Dp,         /* mx1 */
   *diag_jacTjac,   /* diagonal of J^T J, mx1 */
       *pDp;        /* p + Dp, mx1 */

register LM_REAL mu,  /* damping constant */
                tmp; /* mainly used in matrix & vector multiplications */
LM_REAL p_eL2, jacTe_inf, pDp_eL2; /* ||e(p)||_2, ||J^T e||_inf, ||e(p+Dp)||_2 */
LM_REAL p_L2, Dp_L2=LM_REAL_MAX, dF, dL;
LM_REAL tau, eps1, eps2, eps2_sq, eps3;
LM_REAL init_p_eL2;
int nu=2, nu2, stop=0, nfev, njev=0, nlss=0;
const int nm=n*m;
int (*linsolver)(LM_REAL *A, LM_REAL *B, LM_REAL *x, int m)=NULL;

  mu=jacTe_inf=0.0; /* -Wall */
  //残差函数的数量需要大于优化参数的数量
  if(n<m){
    fprintf(stderr, LCAT(LEVMAR_DER, "(): cannot solve a problem with fewer measurements [%d] than unknowns [%d]\n"), n, m);
    return LM_ERROR;
  }
  //调用者需要提供计算残差函数雅可比矩阵的方法，否则请调用带有自动微分的LEVMAR_DIF
  if(!jacf){
    fprintf(stderr, RCAT("No function specified for computing the Jacobian in ", LEVMAR_DER)
        RCAT("().\nIf no such function is available, use ", LEVMAR_DIF) RCAT("() rather than ", LEVMAR_DER) "()\n");
    return LM_ERROR;
  }
  //使用默认或重载的算法超参
  if(opts){
	  tau=opts[0];
	  eps1=opts[1];
	  eps2=opts[2];
	  eps2_sq=opts[2]*opts[2];
    eps3=opts[3];
  }
  else{ // use default values
	  tau=LM_CNST(LM_INIT_MU);
	  eps1=LM_CNST(LM_STOP_THRESH);
	  eps2=LM_CNST(LM_STOP_THRESH);
	  eps2_sq=LM_CNST(LM_STOP_THRESH)*LM_CNST(LM_STOP_THRESH);
    eps3=LM_CNST(LM_STOP_THRESH);
  }
  //分配内存以及设置各个中间变量的指针
  if(!work){
    worksz=LM_DER_WORKSZ(m, n); //2*n+4*m + n*m + m*m;
    work=(LM_REAL *)malloc(worksz*sizeof(LM_REAL)); /* allocate a big chunk in one step */
    if(!work){
      fprintf(stderr, LCAT(LEVMAR_DER, "(): memory allocation request failed\n"));
      return LM_ERROR;
    }
    freework=1;
  }

  /* set up work arrays */
  e=work;                 //残差向量error, nx1向量，即前述r向量
  hx=e + n;               //预测向量hx，nx1向量，其与目标向量x构成r向量
  jacTe=hx + n;           //前述J^T*r，mx1向量
  jac=jacTe + m;          //前述J，mxn矩阵
  jacTjac=jac + nm;       //前述J^T*J，mxm矩阵
  Dp=jacTjac + m*m;       //优化向量p的更新量，mx1向量
  diag_jacTjac=Dp + m;    //前述J^T*J的主对角线，mx1向量
  pDp=diag_jacTjac + m;

  //输入当前的优化向量p，计算预测向量hx
  (*func)(p, hx, m, n, adata); nfev=1;
  
  //输入当前预测向量hx与目标向量x，计算残差向量r
  //LEVMAR_L2NRMXMY函数里有些很有趣的实现，可以看看
  p_eL2=LEVMAR_L2NRMXMY(e, x, hx, n);  

  init_p_eL2=p_eL2;
  //判断残差平方和是否为有限值
  if(!LM_FINITE(p_eL2)) stop=7;

  for(k=0; k<itmax && !stop; ++k){
    //判断上一次更新后（即当前优化向量下）残差平方和是否已满足迭代终止条件
    if(p_eL2<=eps3){ /* error is small */
      stop=6;
      break;
    }

    /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
     * Since J^T J is symmetric, its computation can be sped up by computing
     * only its upper triangular part and copying it to the lower part
     */
    //计算全体残差函数构成的多元向量值函数的雅可比矩阵mxn
    (*jacf)(p, jac, m, n, adata); ++njev;

    /* J^T J, J^T e */
    if(nm<__BLOCKSZ__SQ){ // this is a small problem
      /* J^T*J_ij = \sum_l J^T_il * J_lj = \sum_l J_li * J_lj.
       * Thus, the product J^T J can be computed using an outer loop for
       * l that adds J_li*J_lj to each element ij of the result. Note that
       * with this scheme, the accesses to J and JtJ are always along rows,
       * therefore induces less cache misses compared to the straightforward
       * algorithm for computing the product (i.e., l loop is innermost one).
       * A similar scheme applies to the computation of J^T e.
       * However, for large minimization problems (i.e., involving a large number
       * of unknowns and measurements) for which J/J^T J rows are too large to
       * fit in the L1 cache, even this scheme incures many cache misses. In
       * such cases, a cache-efficient blocking scheme is preferable.
       *
       * Thanks to John Nitao of Lawrence Livermore Lab for pointing out this
       * performance problem.
       *
       * Note that the non-blocking algorithm is faster on small
       * problems since in this case it avoids the overheads of blocking. 
       */

      /* looping downwards saves a few computations */
      register int l;
      register LM_REAL alpha, *jaclm, *jacTjacim;

      for(i=m*m; i-->0; )
        jacTjac[i]=0.0;
      for(i=m; i-->0; )
        jacTe[i]=0.0;

      for(l=n; l-->0; ){
        jaclm=jac+l*m;
        for(i=m; i-->0; ){
          jacTjacim=jacTjac+i*m;
          alpha=jaclm[i]; //jac[l*m+i];
          for(j=i+1; j-->0; ) /* j<=i computes lower triangular part only */
            jacTjacim[j]+=jaclm[j]*alpha; //jacTjac[i*m+j]+=jac[l*m+j]*alpha

          /* J^T e */
          jacTe[i]+=alpha*e[l];
        }
      }

      for(i=m; i-->0; ) /* copy to upper part */
        for(j=i+1; j<m; ++j)
          jacTjac[i*m+j]=jacTjac[j*m+i];

    }
    else{ // this is a large problem
      //利用高效缓存计算J^T*J
      LEVMAR_TRANS_MAT_MAT_MULT(jac, jacTjac, n, m);

      //利用高效缓存计算J^T*e
      for(i=0; i<m; ++i)
        jacTe[i]=0.0;

      for(i=0; i<n; ++i){
        register LM_REAL *jacrow;

        for(l=0, jacrow=jac+i*m, tmp=e[i]; l<m; ++l)
          jacTe[l]+=jacrow[l]*tmp;
      }
    }

	//计算J^T*e的L inf与p的L2，并存储最新的J^T*J的主对角线
    for(i=0, p_L2=jacTe_inf=0.0; i<m; ++i){
      if(jacTe_inf < (tmp=FABS(jacTe[i]))) jacTe_inf=tmp;
      //注意[2]：记录J^T*J主对角线数据，在后续的内层循环中作为记录使用
      diag_jacTjac[i]=jacTjac[i*m+i]; /* save diagonal entries so that augmentation can be later canceled */
      p_L2+=p[i]*p[i];
    }

    //判断J^T*e的L inf是否已满足迭代终止条件
    if((jacTe_inf <= eps1)){
      Dp_L2=0.0; /* no increment for p in this case */
      stop=1;
      break;
    }

    //计算首次迭代时的阻尼系数lambda=tao*tmp
    //tmp为当前J^T*J的主对角线最大值
    if(k==0){
      for(i=0, tmp=LM_REAL_MIN; i<m; ++i)
        if(diag_jacTjac[i]>tmp) tmp=diag_jacTjac[i]; /* find max diagonal element */
      mu=tau*tmp;
    }

    //内层循环，自适应调整阻尼系数对优化向量进行更新
    while(1){
      //注意[2]：更新J^T*J为J^T*J+lambda*I
      for(i=0; i<m; ++i)
        jacTjac[i*m+i]+=mu;

      //求解线性方程得到优化向量的更新量Dp
#ifdef HAVE_LAPACK
      /* 7 alternatives are available: LU, Cholesky + Cholesky with PLASMA, LDLt, 2 variants of QR decomposition and SVD.
       * For matrices with dimensions of at least a few hundreds, the PLASMA implementation of Cholesky is the fastest.
       * From the serial solvers, Cholesky is the fastest but might occasionally be inapplicable due to numerical round-off;
       * QR is slower but more robust; SVD is the slowest but most robust; LU is quite robust but
       * slower than LDLt; LDLt offers a good tradeoff between robustness and speed
       */

      issolved=AX_EQ_B_BK(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_BK;
      //issolved=AX_EQ_B_LU(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_LU;
      //issolved=AX_EQ_B_CHOL(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_CHOL;
#ifdef HAVE_PLASMA
      //issolved=AX_EQ_B_PLASMA_CHOL(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_PLASMA_CHOL;
#endif
      //issolved=AX_EQ_B_QR(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_QR;
      //issolved=AX_EQ_B_QRLS(jacTjac, jacTe, Dp, m, m); ++nlss; linsolver=(int (*)(LM_REAL *A, LM_REAL *B, LM_REAL *x, int m))AX_EQ_B_QRLS;
      //issolved=AX_EQ_B_SVD(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_SVD;

#else
      /* use the LU included with levmar */
      issolved=AX_EQ_B_LU(jacTjac, jacTe, Dp, m); ++nlss; linsolver=AX_EQ_B_LU;
#endif /* HAVE_LAPACK */

      //线性方程求解成功时
      if(issolved){
        //判断优化向量更新量是否满足内层循环跳出条件
        for(i=0, Dp_L2=0.0; i<m; ++i){
          pDp[i]=p[i] + (tmp=Dp[i]);
          Dp_L2+=tmp*tmp;
        }
        if(Dp_L2<=eps2_sq*p_L2){ /* relative change in p is small, stop */
        //if(Dp_L2<=eps2*(p_L2 + eps2)){ /* relative change in p is small, stop */
          stop=2;
          break;
        }
       
       //判断线性方程是否病态，若是，跳出内层循环
       if(Dp_L2>=(p_L2+eps2)/(LM_CNST(EPSILON)*LM_CNST(EPSILON))){ /* almost singular */
         stop=4;
         break;
       }

       //在更新的优化向量下重新计算预测向量hx及残差向量r平方和
       //注意[1]：这里残差向量r平方和的结果保存在hx区域
        (*func)(pDp, hx, m, n, adata); ++nfev; 
        pDp_eL2=LEVMAR_L2NRMXMY(hx, x, hx, n);
       //若新的残差向量r平方和不是有限值，跳出内存循环
        if(!LM_FINITE(pDp_eL2)){
          stop=7;
          break;
        }

        for(i=0, dL=0.0; i<m; ++i)
          dL+=Dp[i]*(mu*Dp[i]+jacTe[i]);
        //目标函数减少量
        dF=p_eL2-pDp_eL2;

        //当前阻尼系数合适，接受优化向量的更新，更新阻尼系数，覆写e区域，覆写游标向量，跳出内存循环
        if(dL>0.0 && dF>0.0){
          tmp=(LM_CNST(2.0)*dF/dL-LM_CNST(1.0));
          tmp=LM_CNST(1.0)-tmp*tmp*tmp;
          mu=mu*( (tmp>=LM_CNST(ONE_THIRD))? tmp : LM_CNST(ONE_THIRD) );
          nu=2;

          for(i=0 ; i<m; ++i) /* update p's estimate */
            p[i]=pDp[i];
          //与注意[1]对应，接受更新的情况下覆写到e区域
          for(i=0; i<n; ++i) /* update e and ||e||_2 */
            e[i]=hx[i];
          p_eL2=pDp_eL2;
          break;
        }
      }

      //不接受当前阻尼系数下的优化向量更新，调整阻尼系数，进入内层循环的下一次迭代
      mu*=nu;//阻尼系数放大nu倍
      //判断阻尼系数放大倍数能否继续增大
      nu2=nu<<1; // 2*nu;
      if(nu2<=nu){ /* nu has wrapped around (overflown). Thanks to Frank Jordan for spotting this case */
        stop=5;
        break;
      }
      //阻尼系数放大倍数翻倍
      nu=nu2;
      //注意[2]：将J^T*J主对角线覆写回到jacTjac区域
      for(i=0; i<m; ++i) /* restore diagonal J^T J entries */
        jacTjac[i*m+i]=diag_jacTjac[i];
    } /* inner loop */
  }

  if(k>=itmax) stop=3;

  for(i=0; i<m; ++i) /* restore diagonal J^T J entries */
    jacTjac[i*m+i]=diag_jacTjac[i];

  if(info){
    info[0]=init_p_eL2;
    info[1]=p_eL2;
    info[2]=jacTe_inf;
    info[3]=Dp_L2;
    for(i=0, tmp=LM_REAL_MIN; i<m; ++i)
      if(tmp<jacTjac[i*m+i]) tmp=jacTjac[i*m+i];
    info[4]=mu/tmp;
    info[5]=(LM_REAL)k;
    info[6]=(LM_REAL)stop;
    info[7]=(LM_REAL)nfev;
    info[8]=(LM_REAL)njev;
    info[9]=(LM_REAL)nlss;
  }

  /* covariance matrix */
  if(covar){
    LEVMAR_COVAR(jacTjac, covar, p_eL2, m, n);
  }

  if(freework) free(work);

#ifdef LINSOLVERS_RETAIN_MEMORY
  if(linsolver) (*linsolver)(NULL, NULL, NULL, 0);
#endif

  return (stop!=4 && stop!=7)?  k : LM_ERROR;
}
```