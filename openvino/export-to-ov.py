"""
Prereqs for running this script:

git clone -b mo_resunet https://github.com/dkurt/openvino_contrib/
source /opt/intel/openvino_2021.4.582/bin/setupvars.sh
export PYTHONPATH=/path/to/openvino_contrib/modules/mo_pytorch/:$PYTHONPATH

"""

from pathlib import Path
import os
import numpy as np
import torch
from BrainMaGe.models.networks import fetch_model

import mo_pytorch
from openvino.inference_engine import IECore

brainmage_root = Path('../')
pytorch_model_path = brainmage_root / 'BrainMaGe/weights/resunet_ma.pt'
#ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp16'
ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp32'

if not os.path.exists(ov_model_dir):
    os.makedirs(ov_model_dir)

pt_model = fetch_model(modelname="resunet", num_channels=1, num_classes=2, num_filters=16)
checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
pt_model.load_state_dict(checkpoint["model_state_dict"])

pt_model.eval()

# Test Accuracy

# Create dummy data
np.random.seed(123)
input_image = torch.Tensor(np.random.standard_normal([1, 1, 128, 128, 128]).astype(np.float32))

print("Running PyTorch Inference on random data...")
# Run Pytorch Inference
with torch.no_grad():
    pt_output = pt_model(input_image)

pt_out = pt_output.detach().numpy()

print("Converting PyTorch model to OpenVINO IR...")

ov_model_name = 'resunet_ma'
model_name = ov_model_dir / ov_model_name

mo_pytorch.convert(pt_model, input_shape=[1, 1, 128, 128, 128], data_type="FP32", model_name=model_name)

print (f"\nOpenVINO model saved at {ov_model_dir} \n")

# Run OpenVINO Inference and Compare Inference results
model_xml = f'{ov_model_dir}/{ov_model_name}.xml'

ie = IECore()
net = ie.read_network(model_xml)
exec_net = ie.load_network(net, 'CPU')
inp_name = next(iter(net.inputs.keys()))
out = exec_net.infer({inp_name: input_image})
out = next(iter(out.values()))

# Print Results.
print('PyTorch output shape:', pt_out.shape)
print('OpenVINO output shape:', out.shape)
print('PyTorch output min and max:', np.min(pt_out), np.max(pt_out))
print('OpenVINO output min and max:', np.min(out), np.max(out))
print('PyTorch and OpenVINO Output difference:', np.max(np.abs(out - pt_out)))

