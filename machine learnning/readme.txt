nn.Conv2d
"""
params
	in_channels: 进通道数,
	out_channels: 出通道数,
	kernel_size: 核大小,
	stride: 步长, stride=1 pointwise
	padding: 填充,默认zeros，先边缘填0，再filter
	dilation: _size_2_t = 1,
	groups: int = 1, in_channels=out_channels=groups Depthwise
	bias: bool = True,
	padding_mode: str = 'zeros'  # TODO: refine this type
"""

ConvTranspose2d
"""
	in_channels: 进通道数,
	out_channels: 出通道数,
	kernel_size: 核大小,
	stride: _步长,
	padding: 输入的每一条边补充0的层数，高宽都增加2*padding, (kernel_size - 1)/2
	output_padding: 输出边补充0的层数，高宽都增加padding,
	groups: 从输入通道到输出通道的阻塞连接数,
	bias: bool = True,
	dilation: 卷积核元素之间的间距,
	padding_mode: str = 'zeros'
"""
原图Height Width=>相邻两列插入 stride-1 的列
Height′=Height+(Stride−1)∗(Height−1)
Width′=Width+(Stride−1)∗(Width)
对Height′ Width′ 做stride=1 kernel_size=kernel_size padding=size-padding-1的卷积 注意padding是两边都padding
ouput=input+(Stride−1)∗(input−1)+2*(kernel_size-padding-1)-kernel_size-1+output_padding
output = (input-1)*stride-2*padding+kernel_size+output_padding

padding = 2
output_padding = 1
kernel_size = 5
stride = 2
刚好使得input翻倍

optimizer.zero_grad()             ## 梯度清零 
preds = model(inputs)             ## inference
loss = criterion(preds, targets)  ## 求解loss
loss.backward()                   ## 反向传播求解梯度
optimizer.step()                  ## 更新权重参数


不进行optimizer.zero_grad()这一步操作，backward()的时候就会累加梯度 
model.zero_grad() optimizer = optim.Optimizer(net.parameters()) 时两者等价