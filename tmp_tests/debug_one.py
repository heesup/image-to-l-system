import os, sys, traceback
sys.path.insert(0, os.getcwd())
import torch
from diffusion_based.dataset.generate_tensor_shards import load_species_xml_samples, encode_sample
from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

samples = load_species_xml_samples('dataset/helios_data', 'cowpea')
print('xml found:', len(samples))
s_info = samples[0]
print('xml:', s_info['xml'])
arr = PlantOrganArray.from_xml_file(s_info['xml'])
renderer = HeliosPyTorchRenderer(image_size=128, device='cuda')
try:
    sample = encode_sample(arr, renderer, torch.device('cuda'), max_slots=4096, use_pyramid=True, dap=s_info['dap'])
    print('OK image:', sample['image'].shape, 'nodes:', sample['nodes'].shape)
except Exception:
    traceback.print_exc()
