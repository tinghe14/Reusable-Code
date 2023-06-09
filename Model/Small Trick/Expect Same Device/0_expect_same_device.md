Error Message: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!

My Experience: to every data move to CUDA, (model, optmizer, data), final calculation can move back to CPU

code to check whether data is in CUDA:
~~~
data.is_cuda
~~~
code to check whether model is in CUDA:
~~~
next(model.parameters()).is_cuda
~~~
