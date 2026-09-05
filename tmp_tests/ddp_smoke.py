import os, torch
import torch.distributed as dist
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.dataset.part_array_dataset import FM_NODE_DIM
m = PartFlowMatchingModel(max_nodes=64, node_dim=FM_NODE_DIM, image_size=128).cuda()
m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[int(os.environ["LOCAL_RANK"])])
x = torch.randn(2, 64, FM_NODE_DIM, device="cuda")
t = torch.rand(2, device="cuda")
img = torch.randn(2, 3, 128, 128, device="cuda")
out = m(x, t, img)["pred_velocity"]
out.sum().backward()
print(f"rank {rank}: DDP fwd/bwd OK, out shape {tuple(out.shape)}")
dist.destroy_process_group()
