#   Intro
* 共享内存的线程模型
  * 每个线程都可以有局部数据，也有共享的全局变量。多个线程共享相同的地址空间，每个线程都能访问全局内存。
  * 线程的栈被分配在堆区，但只有主线程的栈可以动态增长，而子线程的栈是固定的。
  * 线程之间的通信依靠共享的内存实现。
* OpenMP是一套支持跨平台的共享内存方式的多线程并发的编程API
  * fork join模型，parallel region并行区域
  * 线程相关的函数调用库 + 线程相关的编译指导语句 + 线程相关的环境变量
* OpenMP执行特点
  * OpenMP 程序串行和并行区域交替出现
  * 串行区域由 master thread（thread 0） 执行
  * 并行区域由多个线程同一时间一起执行，不同线程一起完成并行区域中的任务
* VS2022只支持到OpenMP2.0
  
# 线程相关的函数调用库
```
omp_set_num_threads(4); // 设置进程的线程数为 4

omp_get_thread_num()；// 获取当前线程的线程号

int mex_thread_num = omp_get_mex_thread(); // 获取最多可以用于并行计算的线程数目

int thread_id = omp_get_thread_num();  // 获取当前线程的 id

int curTime = omp_get_wtime(); // 获取当前时间，秒为单位

int is_in_parallel = omp_in_parallel(); // 当前程序是否在并行中，1 表示并行，0表示串行

omp_set_nested(1); // 设置允许嵌套并行

int is_nested =  omp_get_netsted(); // 获取当前程序是否允许嵌套并行
```

# 常用线程相关的环境变量
* OMP_NUM_THREADS，它指明并行区域中最大的线程数目。
* OMP_SCHEDULE，它指明使用 schedule(runtime) 语句时将采取的调用策略.
* OMP_NESTED，它指明是否允许 OpenMP 采取嵌套并行。

# 编译指导语句
* 编译指导语句在于告知编译器
  * 哪些代码段应该并行进行？
  * 在并行进行的代码段中变量的属性是什么？ 线程公有？线程私有？初始值应该如何确定？
  * 应该采取什么策略调派线程执行并行？
* 格式
`#pragma omp 指令 [子句1] [子句2] [...]`

# 并行区域指令分摊
## parallel指令
* 用在一个结构块之前，表示这段代码将被多个线程并行执行
```
omp_set_num_threads(4);
#pragma omp parallel
{
    //do some things in each thread
}
```

## for指令
* 让多个线程合作完成同一个循环。
* 常见于在并行域（外部嵌套有 parallel）中，可以直接和 parallel 连用，既创建并行区域，又指导线程共同完成循环。
* for指令后需要直接接for循环体，并且循环体三段语句有限制。
* for循环体的循环次数可以小于并行区域的预设线程数。
```
omp_set_num_threads(4);
#pragma omp parallel for
for (int i = 0; i < 10; ++i) 
{
    std::cout << omp_get_thread_num() << std::endl;
}
```

## sections指令
* 划定一块代码区域，在代码区域内指定 section 代码块，不同的线程将会执行不同的代码块（每个代码块只会被一个线程执行），通过这种方式来指导线程进行分工。
* 一般使用在并行域（外部嵌套有 parallel）中，并且可以和 parallel 连用，成为 parallel sections。
```
#pragma omp parallel sections 
{
#pragma omp section 
    {
        for (int i = 0; i < 3; ++i) {
            printf("hhh, here is thread %d and section 1\n", omp_get_thread_num());
        }
    }
#pragma omp section 
    {
        for (int i = 0; i < 3; ++i) {
            printf("hhh, here is thread %d and section 2\n", omp_get_thread_num());
        }
    }
#pragma omp section 
    {
        for (int i = 0; i < 3; ++i) {
            printf("hhh, here is thread %d and section 3\n", omp_get_thread_num());
        }
    }
}
```
* sections 也能写成单独列出，不嵌套在外层 parallel 中，这时候sections 里面的 section 是串行执行的。

## single指令与master指令

## barrier指令

# 并行区域外的变量与并行区域内的同名变量的关系
* 循环构造区域内循环迭代变量以及声明在循环构造区域内的自动变量均为私有变量。

## private子句
* 完全屏蔽并行区域外的同名变量，互不影响（但不用重新声明）。

## firstprivate子句
* 使用并行区域外的变量作为线程同名变量的初始值，但二者相互独立。

## lastprivate子句
* 退出并行区域时将线程变量赋值到并行区域外的同名变量。
* for指令场合，将最后一次执行循环体后的线程变量赋值到并行区域外的同名变量。
* sections指令场合，将程序语法上（而不是实际运行时）最后一个section语句中的线程变量赋值到并行区域外的同名变量。
* 类变量的场合待补充。

## threadprivate
* 待补充

## shared子句

## default子句

## reduction子句

