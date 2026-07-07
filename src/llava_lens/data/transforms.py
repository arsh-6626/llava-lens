from typing import Callable, Optional

from torchvision import transforms


def get_transforms(
    image_size: int = 336,
    augment: bool = False,
    normalize: bool = True,
) -> Callable:
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    return transforms.Compose(transform_list)


def get_test_transforms(image_size: int = 336) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
