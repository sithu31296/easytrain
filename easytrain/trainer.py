import torch
import time
import json
import yaml
import datetime
import shutil
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel

from .dataloaders import *
from .optimizers import *
from .schedulers import *
from .engine import *
from easytrain.utils.distributed import *



class Trainer:
    def __init__(self,
        model: torch.nn.Module,
        trainset: Dataset,
        testset: Dataset,
        criterion: torch.nn.Module,
        config: dict,
        collate_fn = None,
        metric_fn = None,
        model_weights = None,
    ) -> None:
        torch.backends.cudnn.deterministic = True
        fix_seed(42 + get_rank())

        self.config = config
        self.gpu = init_distributed_mode()
        self.num_of_gpus = get_world_size()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trainloader, self.train_sampler = create_train_dataloader(trainset, config['batch_size'], collate_fn)
        self.testloader = create_val_dataloader(testset, config['batch_size'], collate_fn)

        self.model = model
        self.model = self.model.to(self.device)
        self.model_wo_ddp = self.model

        if is_dist_avail_and_initialized():
            self.model = DistributedDataParallel(model, device_ids=[self.gpu], find_unused_parameters=False, output_device=self.gpu)
            self.model_wo_ddp = self.model.module

        self.criterion = criterion
        self.metric_fn = metric_fn
        self.optimizer = create_optimizer(config['optimizer'], self.model.parameters(), config['lr'], config['weight_decay'])
        self.scheduler = create_scheduler(self.optimizer, trainset, self.trainloader, config['batch_size'], config['epochs'], config['lr'], config['end_lr'], config['warmup_iters'])

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
                self.train_sampler.set_epoch(epoch)

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


