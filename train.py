import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import parse_args
from augmentations import simsiam_augmentation, simsiam_transform
from datasets import TileDataset
from models import SimSiam
from optimizers import get_optimizer
from schedulers import get_scheduler


def train(args):
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
    model.to("cuda", non_blocking=True)
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
        for train_step, (tiles1, tiles2) in enumerate(
            tqdm(train_dl, desc=f"Epoch {epoch} / Train Step")
        ):
            tiles1, tiles2 = tiles1.to("cuda"), tiles2.to("cuda")
            train_outputs = model(tiles1, tiles2)
            train_loss = train_outputs["loss"]
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if args.eval_tile_dir:
            model.eval()
            for eval_step, (tiles1, tiles2) in enumerate(
                tqdm(eval_dl, desc=f"Epoch {epoch} / Eval Step")
            ):
                with torch.no_grad():
                    tiles1, tiles2 = tiles1.to("cuda"), tiles2.to("cuda")
                    eval_outputs = model(tiles1, tiles2)
                    eval_loss = eval_outputs["loss"]


if __name__ == "__main__":
    args = parse_args()
    train(args)
