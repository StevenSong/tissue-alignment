import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arguments import parse_args
from augmentations import simsiam_augmentation, simsiam_transform
from datasets import TileDataset
from models import SimSiam
from optimizers import get_optimizer
from schedulers import get_scheduler


def train(args):
    # initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # load datasets
    train_ds = TileDataset(
        name="train",
        tile_dir=args.tile_dir,
        file_ext=args.file_ext,
    )
    mean, std = train_ds.get_mean_std()
    tile_size = train_ds.get_tile_size()
    augmentation = simsiam_augmentation(size=tile_size, mean=mean, std=std)
    transform = simsiam_transform(augmentation=augmentation)
    train_ds.transform = transform
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.eval_tile_dir:
        eval_ds = TileDataset(
            name="eval",
            tile_dir=args.eval_tile_dir,
            file_ext=args.file_ext,
            transform=transform,
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # load model and optimizer
    model = SimSiam(
        backbone=args.backbone,
        projector_hidden_dim=args.projector_hidden_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        output_dim=args.output_dim,
    )
    model.to("cuda")
    optimizer = get_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr * args.batch_size / 256,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
    )

    # training loop
    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        model.train()
        train_losses = []
        for train_step, (tiles1, tiles2) in enumerate(
            tqdm(train_dl, desc=f"Epoch {epoch} / Train Step")
        ):
            tiles1, tiles2 = tiles1.to("cuda"), tiles2.to("cuda")
            if epoch == 0 and train_step == 0:
                writer.add_graph(model, (tiles1, tiles2))
            train_loss = model(tiles1, tiles2)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss = train_loss.detach().to("cpu")
            train_losses.append(train_loss)
            writer.add_scalar(
                "Loss/train/step", train_loss, train_step + epoch * len(train_dl)
            )
        writer.add_scalar("Loss/train/epoch", np.asarray(train_losses).mean(), epoch)

        if args.eval_tile_dir:
            model.eval()
            eval_losses = []
            for eval_step, (tiles1, tiles2) in enumerate(
                tqdm(eval_dl, desc=f"Epoch {epoch} / Eval Step")
            ):
                with torch.no_grad():
                    tiles1, tiles2 = tiles1.to("cuda"), tiles2.to("cuda")
                    eval_loss = model(tiles1, tiles2)
                    eval_loss = eval_loss.detach().to("cpu")
                    eval_losses.append(eval_loss)
                    writer.add_scalar(
                        "Loss/eval/step", eval_loss, eval_step + epoch * len(eval_dl)
                    )
                writer.add_scalar(
                    "Loss/eval/epoch", np.asarray(eval_losses).mean(), epoch
                )

        if epoch + 1 % args.checkpoint_interval == 0:
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
