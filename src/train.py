import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arguments import parse_args
from dataloaders import DATALOADER_T, DATALOADERS
from metrics import METRIC_T, METRICS
from models import MODEL_T, MODELS
from optimizers import OPTIMIZERS
from schedulers import SCHEDULERS


def _train_epoch(
    *,  # enforce kwargs
    epoch: int,
    train_dl: DATALOADER_T,
    model: MODEL_T,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
):
    model.train()
    train_losses = []
    for train_step, train_batch in enumerate(
        tqdm(train_dl, desc=f"Epoch {epoch} / Train Step")
    ):
        train_batch = {k: v.to("cuda") for k, v in train_batch.items()}
        train_outputs = model(train_batch)
        train_loss = train_outputs["loss"]
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss = train_loss.detach().to("cpu")
        train_losses.append(train_loss)
        writer.add_scalar(
            "loss/train/step", train_loss, train_step + epoch * len(train_dl)
        )
        writer.add_scalar(
            "lr/step",
            scheduler.get_last_lr()[0],
            train_step + epoch * len(train_dl),
        )
    writer.add_scalar("loss/train/epoch", np.asarray(train_losses).mean(), epoch)


def _eval_epoch(
    *,  # enforce kwargs
    epoch: int,
    eval_dl: DATALOADER_T,
    model: MODEL_T,
    metrics: List[Tuple[str, METRIC_T]],
    writer: SummaryWriter,
    output_dir: str,
):
    model.eval()
    eval_losses = []
    eval_scores = defaultdict(lambda: defaultdict(list))
    for eval_step, eval_batch in enumerate(
        tqdm(eval_dl, desc=f"Epoch {epoch} / Eval Step")
    ):
        with torch.no_grad():
            eval_batch = {k: v.to("cuda") for k, v in eval_batch.items()}
            eval_outputs = model(eval_batch)
            eval_loss = eval_outputs["loss"]
            eval_loss = eval_loss.detach().to("cpu")
            eval_losses.append(eval_loss)

            for metric, metric_fn in metrics:
                score = metric_fn(
                    model=model,
                    outputs=eval_outputs,
                    batch=eval_batch,
                )
                for k, v in score.items():
                    eval_scores[metric][k].append(v)

    writer.add_scalar("loss/eval/epoch", np.asarray(eval_losses).mean(), epoch)
    for metric, _ in metrics:
        for submetric, scores in eval_scores[metric].items():
            writer.add_scalar(
                f"{metric}/{submetric}/eval/epoch",
                np.asarray(scores).mean(),
                epoch,
            )


def train(args):
    if os.path.exists(args.output_dir):
        raise ValueError(
            f"Output directory for this run already exists: {args.output_dir}"
        )

    # initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # get dataloaders
    get_dataloader = DATALOADERS[args.loader]
    train_dl, eval_dl = get_dataloader(
        data_paths=args.data_paths,
        eval_data_paths=args.eval_data_paths,
        loader_params=args.loader_params,
        eval_loader_params=args.eval_loader_params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # load model
    get_model = MODELS[args.model_arch]
    model = get_model(
        model_params=args.model_params,
    )
    model.to("cuda")

    # get optimizer and scheduler
    get_optimizer = OPTIMIZERS[args.optimizer]
    optimizer = get_optimizer(
        model=model,
        lr=args.lr,
        optimizer_params=args.optimizer_params,
    )
    get_scheduler = SCHEDULERS[args.scheduler]
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        steps_per_epoch=len(train_dl),
        scheduler_params=args.scheduler_params,
    )

    # get evaluation metrics
    metrics = []
    if eval_dl is not None and args.metrics is not None:
        for metric in args.metrics:
            get_metric = METRICS[metric]
            metric_fn = get_metric(metric_params=args.metric_params)
            metrics.append((metric, metric_fn))

    # training loop
    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        _train_epoch(
            epoch=epoch,
            train_dl=train_dl,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=writer,
        )

        if eval_dl is not None:
            _eval_epoch(
                epoch=epoch,
                eval_dl=eval_dl,
                model=model,
                metrics=metrics,
                writer=writer,
                output_dir=args.output_dir,
            )

        if (epoch + 1) % args.checkpoint_interval == 0:
            mpath = os.path.join(args.output_dir, f"checkpoints/{epoch:04}.pt")
            os.makedirs(os.path.dirname(mpath), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                },
                mpath,
            )


if __name__ == "__main__":
    args = parse_args()
    train(args)
