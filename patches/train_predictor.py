import os
import sys
import csv
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import comet_ml

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.
    """
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # pin_memory=False: avoids CUDA abort on Windows with multiprocessing workers
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=config.get('num_workers', 2), pin_memory=False, drop_last=False
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def init_csv_log(save_dir: str, resume: bool) -> str:
    """初始化 CSV loss 記錄檔，resume 時附加寫入，全新時建立標頭。"""
    csv_path = os.path.join(save_dir, 'training_log.csv')
    if not resume or not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['type', 'epoch', 'step', 'loss', 'lr'])
    return csv_path


def append_csv(csv_path: str, row: list):
    """追加一列到 CSV log。"""
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size, start_epoch=0):
    """
    The main training and validation loop for the predictor.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size
        print(f"Effective BATCHSIZE per GPU: {config['batch_size']}, Total: {effective_bs}")

    # CSV log（只在 master process 寫）
    csv_path = None
    if rank == 0:
        csv_path = init_csv_log(save_dir, resume=(start_epoch > 0))

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    # Fast-forward scheduler to match resumed epoch
    if start_epoch > 0:
        fast_forward_steps = start_epoch * len(train_loader)
        if rank == 0:
            print(f"[Resume] Fast-forwarding LR scheduler by {fast_forward_steps} steps...")
        for _ in range(fast_forward_steps):
            scheduler.step()

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = start_epoch * len(train_loader)

    # Check if already fully trained
    if start_epoch >= config['epochs']:
        if rank == 0:
            print(f"[Resume] Predictor already fully trained ({config['epochs']} epochs). Skipping all training.")
        dt_result['best_val_loss'] = best_val_loss
        return dt_result

    for epoch_idx in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
                # CSV: 記錄 train step loss
                if csv_path:
                    append_csv(csv_path, ['train', epoch_idx + 1, batch_idx_global + 1,
                                          round(loss.item(), 6), round(lr, 8)])
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        # Reduce validation metrics
        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            # Always save a per-epoch checkpoint
            epoch_save_path = f"{save_dir}/checkpoints/epoch_{epoch_idx + 1}"
            model.module.save_pretrained(epoch_save_path)
            print(f"Epoch {epoch_idx + 1} checkpoint saved to {epoch_save_path} (Val Loss: {avg_val_loss:.4f})")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_save_path = f"{save_dir}/checkpoints/best_model"
                model.module.save_pretrained(best_save_path)
                print(f"  → New best model also saved to {best_save_path}")

            # CSV: 記錄 epoch val loss
            if csv_path:
                append_csv(csv_path, ['val', epoch_idx + 1, -1, round(avg_val_loss, 6), ''])

            # Write completed epoch count for resume detection
            summary_path = os.path.join(save_dir, 'summary.json')
            _summary_data = {}
            if os.path.isfile(summary_path):
                with open(summary_path) as _f:
                    try:
                        _summary_data = json.load(_f)
                    except Exception:
                        _summary_data = {}
            _summary_data['completed_epochs'] = epoch_idx + 1
            _summary_data['best_val_loss_so_far'] = best_val_loss
            with open(summary_path, 'w') as _f:
                json.dump(_summary_data, _f, indent=4)

        dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    """Main function to orchestrate the DDP training process."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    # Logger and summary setup (master process only)
    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_comet']:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")

    dist.barrier()

    # ── Resume detection ──────────────────────────────────────────
    start_epoch = 0
    summary_path = os.path.join(save_dir, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path) as _f:
            try:
                _s = json.load(_f)
            except Exception:
                _s = {}
        if 'final_result' in _s:
            start_epoch = config['epochs']  # fully done
        else:
            start_epoch = _s.get('completed_epochs', 0)

    # ── Model loading ─────────────────────────────────────────────
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)

    if start_epoch >= config['epochs']:
        if rank == 0:
            print(f"[Resume] Predictor already fully trained ({config['epochs']} epochs). Skipping.")
        cleanup_ddp()
        return

    if start_epoch > 0:
        resume_path = os.path.join(save_dir, 'checkpoints', f'epoch_{start_epoch}')
        if rank == 0:
            print(f"[Resume] Loading from {resume_path}, resuming from Epoch {start_epoch + 1}.")
        model = Kronos.from_pretrained(resume_path)
    else:
        model = Kronos.from_pretrained(config['pretrained_predictor_path'])

    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Predictor Model Size: {get_model_size(model.module)}")

    # Start Training
    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size,
        start_epoch=start_epoch
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(summary_path, 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with `torchrun`.")

    config_instance = Config()
    main(config_instance.__dict__)
