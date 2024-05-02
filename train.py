import functools
import itertools
import os

import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from data import ProcessSupervisedDataset, torch_default_data_collator


def get_policies():
    """Get the policies for mixed precision and fsdp wrapping"""

    mp_wrapper = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    wrapping_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            GPT2Block,
        },
    )
    return mp_wrapper, wrapping_policy


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_entrypoint(rank, world_size, config: DictConfig):
    torch.cuda.set_device(rank)
    if config.train.fsdp:
        setup(rank, world_size)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        device_map="auto" if config.train.fsdp else {"": rank},
    )

    # shard with FSDP
    if config.train.fsdp:
        mp_policy, auto_wrap_policy = get_policies()
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            limit_all_gathers=True,
            sharding_strategy="FULL_SHARD",
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

    if config.train.mode == "ft":
        dataset = ProcessSupervisedDataset(
            config.model.name_or_path,
            config.data.data_file,
            config.data.limit,
            config.data.max_length,
        )
        gen = torch.Generator().manual_seed(config.seed)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [1 - config.data.test_split, config.data.test_split], generator=gen
        )
    else:
        raise NotImplementedError

    if config.train.fsdp:
        train_sampler = DistributedSampler(
            train_ds, rank=rank, num_replicas=world_size, shuffle=True
        )
        test_sampler = DistributedSampler(val_ds, rank=rank, num_replicas=world_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.train_micro_bsz,
        batch_sampler=train_sampler if config.train.fsdp else None,
        shuffle=not config.train.fsdp,
        num_workers=2,
        pin_memory=True,
        collate_fn=torch_default_data_collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.val_micro_bsz,
        batch_sampler=test_sampler if config.train.fsdp else None,
        num_workers=2,
        pin_memory=True,
        collate_fn=torch_default_data_collator,
    )

    # setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.train.warmup_steps,
        num_training_steps=config.train.steps // config.train.grad_acc_steps,
    )

    train_iter = itertools.cycle(train_loader)

    # train loop
    model.train()
    for step, batch in enumerate(train_iter):
        train_step(config, rank, step, batch, model, optimizer, scheduler)

        if step % config.train.eval_interval == 0:
            with torch.no_grad():
                model.eval()
                if rank == 0:
                    print("running eval")
                # evaluate on entire val set
                val_loss = torch.tensor(0.0, device=rank)
                with torch.no_grad():
                    for val_batch in iter(val_loader):
                        val_batch = {k: v.to(rank) for k, v in val_batch.items()}
                        loss = model(**val_batch).loss
                        val_loss += loss

                if dist.is_initialized():
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)

                if rank == 0:
                    print(
                        f"step: {step}, avg_loss: {val_loss.cpu().item() / len(val_loader):.3f}"
                    )
        if step % config.train.save_interval == 0:
            save_model_checkpoint(config, step, model, rank)

        if step == config.train.steps:
            if rank == 0:
                print("training done")
            break


def save_model_checkpoint(
    config: DictConfig,
    step,
    model,
    rank,
):
    if not config.train.do_save:
        return
    """saving model via rank0 cpu streaming and full_state_dict"""
    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    if config.train.fsdp:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state = model.state_dict()
            print(f"saving process: rank {rank}  done w model state_dict\n")
    else:
        cpu_state = model.state_dict()

    if rank == 0:
        print("--> saving model ...")
        save_full_path = os.path.join(
            config.train.save_dir, "policy_step_{}.pt".format(step)
        )
        torch.save(dict(step=step, model=cpu_state), save_full_path)


def train_step(config: DictConfig, rank, step, batch, model, optimizer, scheduler):
    # move batch to device
    batch = {k: v.to(rank) for k, v in batch.items()}
    loss = model(**batch).loss / config.train.grad_acc_steps

    loss.backward()

    if (step + 1) % config.train.grad_acc_steps == 0 or step + 1 == config.train.steps:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.train.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

    metrics = dict(
        step=step,
        loss=loss.detach().cpu().item(),
        grad_norm=grad_norm.item(),
        lr=optimizer.param_groups[0]["lr"],
    )

    if rank == 0 and step % config.train.log_interval == 0:
        print(metrics)
        if wandb.run:
            wandb.log(metrics)
