# 4.1 一个检测CUDA运行时错误的宏函数
* 实现一个宏函数CHECK对runtime API返回错误代号进行检查，建议使用do-while(0)结构。
* 宏函数中使用__FILE__与__LINE__方便定位。
* 一个不建议使用该宏函数的runtime API：cudaEventQuery()，其可能返回cudaErrorNotReady，但不代表程序出错。
* 使用该宏检查核函数的话，可以在调用核函数后使用如下两条语句 —— CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());

# 4.2 用CUDA-MEMCHECK检查内存错误
待补充