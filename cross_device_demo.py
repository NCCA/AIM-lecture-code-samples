import torch

input = torch.rand(1,28,28)

model = torch.nn.Conv1d(28,256,3).to('cuda')
model(input)

# Error
# >> Traceback (most recent call last):
# >>   File "cross_device_demo.py", line 7, in <module>
# >>     model(input)
# >>   File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
# >>     return forward_call(*input, **kwargs)
# >>   File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 302, in forward
# >>     return self._conv_forward(input, self.weight, self.bias)
# >>   File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 298, in _conv_forward
# >>     return F.conv1d(input, weight, bias, self.stride,
# >> RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument weight in method wrapper___slow_conv2d_forward)
