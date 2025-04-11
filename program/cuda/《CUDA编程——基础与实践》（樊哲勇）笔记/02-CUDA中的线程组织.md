# 2.1 略

# 2.2
* 扩展名.cu
* CUDA代码编译器nvcc
* 主机对设备的调用通过核函数实现，核函数必须被限定词 __global__ 修饰，返回类型必须为void。
* 核函数调用时在函数名与()之间有<<<>>>，<<<>>>中指定网格尺寸grid_size与线程块尺寸block_size。
  
# 2.3 CUDA中的线程组织
* 在核函数的内部，配置参数grid_size与block_size是已知的，都是dim3数据，分别限定了block_idx与thread_idx（uint3数据）的取值范围。
* 对thread_idx，x维度变化最快，z维度变化最慢，但block_idx的xyz三个维度是独立的，因为xyz共同决定一个block，所有block独立执行。
* tid与bid的通常定义
* 内建变量线程束大小warpSize，目前所有GPU架构该数值为32。
 * 从开普勒架构开始，xyz三个方向上的乘积上限为1024（即一个线程块中最多只能由1024个线程），xyz三个方向上最大允许的网格大小是2\^31-1，65535，65535。

# 2.4 CUDA中的头文件
略

# 2.5 用nvcc编译CUDA程序
待补充