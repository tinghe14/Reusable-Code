Error Message: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!

My Experience: to every data move to CUDA, (weights from model, optmizer, data), final calculation can move back to CPU
