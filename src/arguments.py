import argparse

from dataloaders import DATALOADERS
from metrics import METRICS
from models import MODELS
from optimizers import OPTIMIZERS
from schedulers import SCHEDULERS
from utils import PARAMS


class KeyValuePairsToDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        args: PARAMS = dict()
        for value in values:
            key, value = value.split("=")
            try:
                value = int(value)
            except ValueError as e1:
                try:
                    value = float(value)
                except ValueError as e2:
                    pass

            if key in args:
                # convert to list
                if isinstance(args[key], list):
                    args[key].append(value)
                else:
                    args[key] = [args[key], value]
            else:
                args[key] = value
        setattr(namespace, self.dest, args)


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
        "--data_paths",
        nargs="*",
        required=True,
        help="Paths to data for data loader.",
    )
    parser.add_argument(
        "--eval_data_paths",
        nargs="*",
        help="Paths to evaluation data. Leave blank to skip evaluation.",
    )
    parser.add_argument(
        "--loader",
        type=str,
        required=True,
        choices=sorted(DATALOADERS.keys()),
        help="Name of data loading module to use.",
    )
    parser.add_argument(
        "--loader_params",
        nargs="*",  # gets parsed to dict
        action=KeyValuePairsToDict,
        metavar="PARAM=VAL",
        help=(
            "Parameters for data loader. "
            "Specified as a list of key-value pairs. "
            "If a key is used more than once, its values will be parsed as a list. ",
            "Values will be parsed as float if possible, string otherwise.",
        ),
    )
    parser.add_argument(
        "--eval_loader_params",
        nargs="*",  # gets parsed to dict
        action=KeyValuePairsToDict,
        metavar="PARAM=VAL",
        help=(
            "Parameters for evaluation data loader. "
            "Specified as a list of key-value pairs. "
            "If a key is used more than once, its values will be parsed as a list. ",
            "Values will be parsed as float if possible, string otherwise.",
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of multiprocessing workers to use for data loading.",
    )

    # MODEL ARGS
    parser.add_argument(
        "--model_arch",
        type=str,
        required=True,
        choices=sorted(MODELS.keys()),
        help="Name of model architecture. Determines modeling strategy.",
    )
    parser.add_argument(
        "--model_params",
        nargs="*",  # gets parsed to dict
        action=KeyValuePairsToDict,
        metavar="PARAM=VAL",
        help=(
            "Parameters for model initialization. "
            "Specified as a list of key-value pairs. "
            "If a key is used more than once, its values will be parsed as a list. ",
            "Values will be parsed as float if possible, string otherwise.",
        ),
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
        help="Base learning rate or learning rate scheduler.",
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
        choices=sorted(OPTIMIZERS.keys()),
        required=True,
        help="Optimizer for training.",
    )
    parser.add_argument(
        "--optimizer_params",
        nargs="*",  # gets parsed to dict
        action=KeyValuePairsToDict,
        metavar="PARAM=VAL",
        help=(
            "Parameters for optimizer. "
            "Specified as a list of key-value pairs. "
            "If a key is used more than once, its values will be parsed as a list. ",
            "Values will be parsed as float if possible, string otherwise.",
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=sorted(SCHEDULERS.keys()),
        required=True,
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--scheduler_params",
        nargs="*",  # gets parsed to dict
        action=KeyValuePairsToDict,
        metavar="PARAM=VAL",
        help=(
            "Parameters for learning rate scheduler. "
            "Specified as a list of key-value pairs. "
            "If a key is used more than once, its values will be parsed as a list. ",
            "Values will be parsed as float if possible, string otherwise.",
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=sorted(METRICS.keys()),
        help=(
            "Evaluation metrics. Only run if evaluation data is provided. "
            "Evaluation loss is always run. Specify multiple as a list of metric names."
        ),
    )
    parser.add_argument(
        "--metric_params",
        nargs="*",  # gets parsed to dict
        action=KeyValuePairsToDict,
        metavar="METRIC.PARAM=VAL",
        help=(
            "Parameters for metrics. "
            "Specified as a list of key-value pairs. "
            "If a key is used more than once, its values will be parsed as a list. ",
            "Parameter key should be prefixed with metric name and dot. "
            "Values will be parsed as float if possible, string otherwise.",
        ),
    )

    args = parser.parse_args()
    return args
