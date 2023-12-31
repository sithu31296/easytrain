import torch
import time
import json
import yaml
import datetime
import shutil
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel

from easytrain.dataloaders import create_dataloader
from easytrain.optimizers import create_optimizer
from easytrain.schedulers import create_scheduler
from easytrain.engine import train_one_epoch, evaluate_one_epoch
from easytrain.utils.torch_utils import setup_cudnn, fix_seed
from easytrain.utils.distributed import *



class Trainer:
    def __init__(self,
        config: dict,
        model: torch.nn.Module,
        trainset: Dataset,
        testset: Dataset,
        criterion: torch.nn.Module,
        collate_fn = None,
        metric_fn = None,
    ) -> None:
        setup_cudnn()
        fix_seed(123 + get_rank())

        self.config = config
        self.gpu = init_distributed_mode()
        self.num_of_gpus = get_world_size()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trainloader, self.testloader = create_dataloader(trainset, testset, config['batch_size'], collate_fn, config['workers'])

        self.model = model
        self.model = self.model.to(self.device)
        self.model_wo_ddp = self.model

        if is_dist_avail_and_initialized():
            self.model = DistributedDataParallel(model, device_ids=[self.gpu], find_unused_parameters=False, output_device=self.gpu)
            self.model_wo_ddp = self.model.module

        self.criterion = criterion
        self.metric_fn = metric_fn
        self.optimizer = create_optimizer(config['optimizer'], self.model, config['lr'], config['weight_decay'])
        self.scheduler = create_scheduler(config['scheduler'], self.optimizer, self.trainloader, config['batch_size'], config['epochs'], config['lr'], warmup_iters=config['warmup_iters'], milestones=config['milestones'])

        self.save_dir = Path(config['save_dir'])

        if is_main_process():
            if self.save_dir.exists() and self.save_dir.is_dir():
                shutil.rmtree(self.save_dir)
            
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.save_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)


    def train(self,
        train_fn = train_one_epoch,
        eval_fn = evaluate_one_epoch,
    ):
        best_score = 0.
        start_time = time.time()
        for epoch in range(self.config['epochs']):
            if is_dist_avail_and_initialized():
                self.trainloader.batch_sampler.sampler.set_epoch(epoch)

            train_stats = train_fn(self.model, self.criterion, self.trainloader, self.optimizer, self.scheduler, self.device, epoch)
            test_stats, current_score = eval_fn(self.model, self.criterion, self.testloader, self.device, self.metric_fn)

            save_on_master(self.model_wo_ddp.state_dict(), str(self.save_dir / "last.pth"))

            if current_score > best_score:
                save_on_master(self.model_wo_ddp.state_dict(), str(self.save_dir / "best.pth"))
                best_score = current_score
            
            log_stats = {"Score": current_score, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"test_{k}": v for k, v in test_stats.items()}, "epoch": epoch}

            if is_main_process():
                with open(self.save_dir / "log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")


