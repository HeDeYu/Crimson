# CSPNeXt
## Cross Stage Partial Network
* base layer划分为2部分，P1直连，P2经过dense block。
* 三种方式结合P1&P2
  * fusion last:cat(P1, conv1x1(F(P2)))
  * fusion first:conv1x1(cat(P1,F(P2)))
  * CSP:conv1x1(cat(P1,conv1x1(F(P2))))

## structure diagram and implement from mmyolo
```
标准卷积模块 ConvModule=conv+norm+activation
深度分离卷积模块

block模块：若干个卷积模块结合残差链接等方式得到的网络
CSPNeXtBlock
DarknetBottleneck

deepen_factor控制中间张量的通道数，基准通道数=64。
widen_factor控制每个stage中的block数量，基准block数量=3。

# in_channels, out_channels, num_blocks, add_identity, use_spp
arch_settings = {
    'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 6, True, False], [512, 1024, 3, False, True]],
    'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
            [256, 512, 6, True, False], [512, 768, 3, True, False],
            [768, 1024, 3, False, True]]
}


Backbone model structure diagram
     +-----------+
     |   input   |
     +-----------+
     |     v     |
     +-----------+
     |   stem    |    self.stem = self.build_stem_layer()
     |           |    self.layers = ['stem']
     |           |    3个卷积模块，第一个卷积模块stride=2，升通道数到stage1输入通道数一半，
     |           |    第二个卷积模块保持通道数，
     |   layer   |    第三个卷积模块升通道数到stage1输入通道数。
     +-----------+
     |     v     |
     +-----------+    for idx, setting in enumerate(arch_setting):
     |   stage   |        stage = []
     |  layer 1  |        stage += self.build_stage_layer(idx, setting)
     +-----------+        self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
     |     v     |        self.layers.append(f'stage{idx + 1}')
     +-----------+
     |   stage   |    每个stage先通过一个卷积stride=2，通道数保持，再进入CSPLayer。
     |  layer 2  |    CSPLayer中
     |           |    short分支为1x1卷积压缩通道数到输出通道数*系数（default=0.5）；
     +-----------+    main分支使用1x1卷积压缩通道数到输出通道数*系数，
     |     v     |    再经过若干个保持通道数的block模块，再与short分支cat。
        ......   
     |     v     |    若有注意力机制，对cat结果执行。
     +-----------+    最后对cat结果执行通道数转为输出通道数的1x1卷积模块。
     |   stage   |
     |  layer n  |
     +-----------+
     In P5 model, n=4 (total stride=32)
     In P6 model, n=5 (total stride=64)
```

# PAFPN
* PAFPN=FPN+PAN
* FPN=feature pyramid network=backbone的多个尺度输出都用上，top layer的信息融合到down layer中。
* PAN=path aggreation network=FPN的多个尺度输出再采用top-down融合一遍才输出到head
## structure diagram and implement from mmyolo
```

     P5 neck model structure diagram
                        +--------+                     +-------+
                        |top_down|----------+--------->|  out  |---> output0
                        | layer1 |          |          | layer0|
                     |CSP256*2->256|        |
                        +--------+          |          +-------+
                             ^              |
     stride=8                |              |
     C=256                 C=512
     idx=0  +------+    +--------+          |
     -----> |reduce|--->|   cat  |          |
            |layer0|    +--------+          |
            |  I   |         ^              |
            +------+         |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer1 |    |  layer0   |
                        +--------+    +-----------+
                             ^              |
                             |              |
                           C=256            |
                    +-------------+         v
                    |   top_down  |    +-----------+
                    |   layer2    |--->|   cat     |
                    |CSP512*2->512|
                    |Conv512->256 |
                    +-------------+    +-----------+
     stride=16               ^              |
     C=512                   |              v
     idx=1  +------+       C=1024     +-----------+    +-------+
     -----> |reduce|--->+--------+    | bottom_up |--->|  out  |---> output1
            |layer1|    |   cat  |    |   layer0  |    | layer1|
            |  I   |    +--------+
            +------+         ^        +-----------+    +-------+
                             |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer2 |    |  layer1   |
     stride=32          +--------+    +-----------+
     C=1024                  |              |
     idx=2  +------+         |              v
     -----> |reduce|         |        +-----------+
            |layer2|-------C=512----->|    cat    |
        |Conv1024->512 |              |           |
            +------+                  +-----------+
                                            v
                                      +-----------+    +-------+
                                      | bottom_up |--->|  out  |---> output2
                                      |  layer1   |    | layer2|
                                      +-----------+    +-------+


     P6 neck model structure diagram
                        +--------+                     +-------+
                        |top_down|----------+--------->|  out  |---> output0
                        | layer1 |          |          | layer0|
                     |CSP256*2->256|        |
                        +--------+          |          +-------+
     stride=8                ^              |
     C=256                   |              |
     idx=0  +------+       C=512
     -----> |reduce|    +--------+          |
            |layer0|--->|   cat  |          |
            |  I   |    +--------+          |
            +------+         ^              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer1 |    |  layer0   |
                        +--------+    +-----------+
                             ^              |
                        +--------+          v
                        |top_down|    +-----------+
                        | layer2 |--->|    cat    |
                    |CSP512*2->512|
                    |Conv512->256 |
                        +--------+    +-----------+
     stride=16               ^              |
     C=512                   |              v
     idx=1  +------+      C=1024
     -----> |reduce|--->+--------+    +-----------+    +-------+
            |layer1|    |   cat  |    | bottom_up |--->|  out  |---> output1
            |  I   |    +--------+    |   layer0  |    | layer1|
            +------+         ^        +-----------+    +-------+
                             |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer2 |    |  layer1   |
                        +--------+    +-----------+
                             ^              |
                        +--------+          v
                        |top_down|    +-----------+
                        | layer3 |--->|    cat    |
                    |CSP768*2->768|
                    |Conv768->512 |
                        +--------+    +-----------+
     stride=32               ^              |
     C=768                   |              |
     idx=2  +------+      C=1536            v
     -----> |reduce|--->+--------+    +-----------+    +-------+
            |layer2|    |   cat  |    | bottom_up |--->|  out  |---> output2
            |  I   |    +--------+    |   layer1  |    | layer2|
            +------+         ^        +-----------+    +-------+
                             |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer3 |    |  layer2   |
                        +--------+    +-----------+
     stride=64               ^              |
     C=1024                  |              v
     idx=3  +------+         |        +-----------+
     -----> |reduce|-------C=768----->|    cat    |
            |layer3|                  +-----------+
        |Conv1024->768|
            +------+                        v
                                      +-----------+    +-------+
                                      | bottom_up |--->|  out  |---> output3
                                      |  layer2   |    | layer3|
                                      +-----------+    +-------+
```

* 顺序构建reduce layer。最顶层(stride最大)的输出接的reduce layer为1x1卷积模块，输出通道数是此顶层输入的通道数(因为没有对应的top down layer进行通道转换)，其它的reduce layer为Identity()。
```
self.reduce_layers = nn.ModuleList()
    for idx in range(len(in_channels)):
        self.reduce_layers.append(self.build_reduce_layer(idx))
```

* 逆序构建upsample layer与top down layer，注意这两类layer的数量比输入张量数量少1（构造时idx=0不会调用）。其中upsample layer默认使用最近邻插值上采样模块。top down layer主体为CSP模块，除了最后一个（对应最底层）外再额外添加1x1卷积模块进行通道适配。
```
self.upsample_layers = nn.ModuleList()
    self.top_down_layers = nn.ModuleList()
    for idx in range(len(in_channels) - 1, 0, -1):
        self.upsample_layers.append(self.build_upsample_layer(idx))
        self.top_down_layers.append(self.build_top_down_layer(idx))
```

* 顺序构建down sample layer与bottom up layer，注意这两类layer的数量比输入张量数量少1（构造时idx=L-1不会被调用）。其中dosample layer使用stride=2的卷积模块。bottom up layer为CSP模块。
```
self.downsample_layers = nn.ModuleList()
    self.bottom_up_layers = nn.ModuleList()
    for idx in range(len(in_channels) - 1):
        self.downsample_layers.append(self.build_downsample_layer(idx))
        self.bottom_up_layers.append(self.build_bottom_up_layer(idx))
```

* 顺序构建out layer为卷积模块。
```
self.out_layers = nn.ModuleList()
    for idx in range(len(in_channels)):
        self.out_layers.append(self.build_out_layer(idx))
``` 