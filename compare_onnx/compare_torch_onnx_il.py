import os
import time
import sys

import numpy as np
import logging
from tqdm import tqdm

from BrainMaGe.models.networks import fetch_model

import onnx
import torch

print(torch.__version__)

from onnx import numpy_helper

IMAGE_SIZE=128

MODEL = "resunet_ma"

TORCH_LAYER_NAME = "ins"
ONNX_LAYER_NAME = "201"

activation = {}


def get_activation(name):
    def hook(model, input, output):
        print("Input shape is: ", len(input), input[0].shape)

        mean = input[0].mean(dim=1)
        var = input[0].var(dim=1)
        #print("Mean is: ", mean)
        #print("Var is: ", var)
        activation[name] = output.detach()
    return hook

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main_torch(model_name):
    model = fetch_model(
        modelname="resunet", num_channels=1, num_classes=2, num_filters=16
    )

    checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device('cpu'))
    model.eval()

    images_out = []
    if MODEL == "resunet_multi_4":
        channel = 4
    else:
        channel = 1

    images = np.random.rand(1, channel, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    bs, c, h,w,d = images.shape
    
    t_im = torch.from_numpy(images) # np.transpose(images, (0,3,1,2)))
    t_im = t_im.type(torch.FloatTensor)    

    model.ins.register_forward_hook(get_activation(TORCH_LAYER_NAME))

    torch_out = model(t_im)
    
    ds_0_output = activation[TORCH_LAYER_NAME]

    #print(ds_0_output)

    out_mask = torch_out.detach().numpy()
    print("output shape is: ", out_mask.shape)

    # Export the model
    torch.onnx.export(model,                               # model being run
                      t_im,                                # model input (or a tuple for multiple inputs)
                      MODEL+"_"+str(IMAGE_SIZE)+".onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,                  # store the trained parameter weights inside the model file
                      # verbose=True,
                      opset_version=12,                    # the ONNX version to export the model to
                      #do_constant_folding=True,            # whether to execute constant folding for optimization
                      input_names = ['input'],             # the model's input names
                      output_names = ['output'],           # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                  'output' : {0 : 'batch_size'}})

    import onnxruntime

    model_path = MODEL+"_"+str(IMAGE_SIZE) + "_" + ONNX_LAYER_NAME + ".onnx"
    model = onnx.load(MODEL+"_"+str(IMAGE_SIZE)+".onnx")

    INTIALIZERS  = model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = W
    
    print(onnx_weights.keys())
    #print(model.graph.node)
    intermediate_tensor_name = ONNX_LAYER_NAME
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    print(intermediate_layer_value_info)
    intermediate_layer_value_info.name = intermediate_tensor_name
    model.graph.output.extend([intermediate_layer_value_info])
    onnx.save(model, model_path)
    ort_session = onnxruntime.InferenceSession(model_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(t_im)}
    ort_outs = ort_session.run(None, ort_inputs)

    #  compare ONNX Runtime and PyTorch results
    print("Torch output shape is: ", ds_0_output.shape)
    print("Onnx output shape is: ", len(ort_outs), ort_outs[0].shape, ort_outs[1].shape)
    np.testing.assert_allclose(ds_0_output, ort_outs[1], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime for Pytorch layer " + TORCH_LAYER_NAME + " and ONNX layer " + ONNX_LAYER_NAME + ", and the result looks good!")
    return out_mask

out_mask = main_torch("../BrainMaGe/weights/" + MODEL + ".pt") 

