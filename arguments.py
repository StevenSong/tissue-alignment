import argparse
import os

from optimizers import OPTIMIZERS
from schedulers import SCHEDULERS


def parse_args():
    parser = argparse.ArgumentParser()

    # RUN ARGS
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder to store results in.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        required=True,
        help="Number of epochs between checkpoints.",
    )

    # DATA ARGS
    parser.add_argument(
        "--tile_dir",
        type=str,
        required=True,
        nargs="+",
        help="Folder(s) containing training image tiles.",
    )
    parser.add_argument(
        "--eval_tile_dir",
        type=str,
        nargs="*",
        help="Folder(s) containing evaluation image tiles. Leave blank to skip evaluation.",
    )
    parser.add_argument(
        "--file_ext",
        type=str,
        default=".png",
        help="Image tile file extension.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of multiprocessing workers to use for data loading.",
    )

    # MODEL ARGS
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Backbone model of encoder.",
    )
    parser.add_argument(
        "--projector_hidden_dim",
        type=int,
        required=True,
        help="Hidden dimension of projection head of encoder.",
    )
    parser.add_argument(
        "--predictor_hidden_dim",
        type=int,
        required=True,
        help="Hidden dimension of prediction head.",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        required=True,
        help="Output dimension of encoder and prediction head.",
    )

    # TRAINING ARGS
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Base learning rate. Scaled by batch_size / 256 as initial learning rate for scheduler.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=OPTIMIZERS,
        required=True,
        help="Optimizer for training. Parameterize with other arguments.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        required=True,
        help="Optimizer momentum parameter. Only used by optimizers which use momentum.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        help="Optimizer weight decay parameter. Only used by optimizers which use weight decay.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=SCHEDULERS,
        required=True,
        help="Learning rate scheduler. Parameterize with other arguments.",
    )

    args = parser.parse_args()
    return args
