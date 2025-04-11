# 5.1 用CUDA事件计时
* 声明CUDA事件类型变量start，stop —— cudaEvent_t。
* 初始化CUDA事件类型变量 —— cudaEventCreate(cudaEvent_t & cuda_event);
* 在需要计时的代码块之前记录一个代表开始的事件 —— cudaEventRecord(cudaEvent_t start);
* 对于WDDM驱动模式的GPU，需要调用cudaEventQuery(start);
* 在需要计时的代码块之后记录一个代表结束的事件 —— cudaEventRecord(cudaEvent_t stop);
* 调用cudaEventSynchronize(stop);让主机等待事件stop被记录完毕。
* 调用cudaEventElapsedTime(float & elapsed_time_ms, cudaEvent_t start, cudaEvent_t stop);获取两个事件的时间差（单位为ms）。
* 销毁CUDA事件类型变量 —— cudaEventDestroy(cudaEvent_t cuda_event);
* 有效显存带宽

# 5.2 几个影响GPU加速的关键因素
* 数据传输的比例
* 算术强度=算术操作的工作量/必要的内存操作
* GPU数据存期比浮点数计算慢。
* 并行规模，可用GPU中总的线程数目来衡量。一个GPU有多个流处理器SM构成，每个SM有若干CUDA核心，一个SM最多驻留的线程个数是2048（对图灵结构，则为1024）。
* 若核函数中定义的线程数目远小于并行规模的话，很难得到很高的加速比。