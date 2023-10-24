import os
import torch
from torch import distributed as dist



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    "total gpu"
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_rank():
    "current gpu"
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    "Disables printing when not in master process"
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    # world_size = num_of_gpu
    # rank = current gpu
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        local_rank = global_rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        return 0
    
    torch.cuda.set_device(local_rank)
    print(f"| distributed init (rank {local_rank})", flush=True)
    dist.init_process_group("nccl")
    dist.barrier()
    setup_for_distributed(global_rank == 0)
    return local_rank


def cleanup():
    dist.destroy_process_group()

def all_gather(data):
    """Run all_gather on arbitary pickable data (not necessarily tensors)
    Args:
        data: any pickable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """Reduce the values in the dictionary from all processes so that all processes have the averaged results.
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Returns:
        a dict with the same fields as input_dict, after reduction
    """
    world_size = get_world_size()
    if world_size == 1:
        return input_dict
    
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processses
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        reduce_dict = {k: v for k, v in zip(names, values)}
    return reduce_dict


def reduce_tensor(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    with torch.inference_mode():
        dist.all_reduce(tensor)
    return tensor