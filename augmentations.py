from typing import Tuple, Callable, Union
import torchvision.transforms as T


def simsiam_transform(*, augmentation: Callable) -> Callable:
    def transform(x):
        x1 = augmentation(x)
        x2 = augmentation(x)
        return x1, x2
    return transform


def simsiam_augmentation(
    *,
    size: Union[int, Tuple[int, int]],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Callable:
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.2, 1.0), antialias=True),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=mean, std=std),
    ])
