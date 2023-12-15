import numpy as np
import matplotlib.pyplot as plt
import torch

import models as M

import torchvision

import resnet_decoder as RD

#encoder = M.Encoder()
encoder = torchvision.models.resnet50()
encoder.fc = torch.nn.Identity()
decoder = M.Decoder()
model = M.Autoencoder_conv(encoder, decoder)


x = torch.randn(1, 3, 512, 432, requires_grad=True)
torch_out = model(x)

exp = "encoder_resnet152"

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"./onnx_models/{exp}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  #opset_version=1,          #the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})