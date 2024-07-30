# %%
import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from models.crater_model import CraterModel
from torch.utils.data import DataLoader
from utils.dataset import CraterDataset

# %%
IMG_DIM = 128

# %%
train_dataset = CraterDataset(
    img_dir=f"./dataset/geo_train_{IMG_DIM}",
    mask_dir=f"./dataset/mask_train_{IMG_DIM}",
)

val_dataset = CraterDataset(
    img_dir=f"./dataset/geo_val_{IMG_DIM}",
    mask_dir=f"./dataset/mask_val_{IMG_DIM}",
)
test_dataset = CraterDataset(
    img_dir=f"./dataset/geo_test_{IMG_DIM}",
    mask_dir=f"./dataset/mask_test_{IMG_DIM}",
)

n_cpus = os.cpu_count()

train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=n_cpus
)
val_dataloader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=n_cpus
)
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=n_cpus
)

# %%
# Visualize dataset

batch = next(iter(train_dataloader))

for image, mask in zip(batch["image"], batch["mask"]):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.numpy().squeeze())
    plt.title("Mask")


# %%
def visualize(model):
    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze())
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze())
        plt.title("Prediction")
        plt.axis("off")

        plt.show()


# %%
fpn_dice = CraterModel(
    arch="FPN",
    encoder_name="resnet34",
    in_channels=3,
    out_classes=1,
    loss_fn=smp.losses.DiceLoss(
        mode=smp.losses.constants.BINARY_MODE, from_logits=True
    ),
    lr=2e-4,
)

logger = TensorBoardLogger("logs", name="crater_fpn_dice", log_graph=True)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
)

trainer.fit(
    fpn_dice,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
visualize(fpn_dice)

# %%
unet_dice = CraterModel(
    arch="Unet",
    encoder_name="resnet34",
    in_channels=3,
    out_classes=1,
    loss_fn=smp.losses.DiceLoss(
        mode=smp.losses.constants.BINARY_MODE, from_logits=True
    ),
    lr=2e-4,
)

logger = TensorBoardLogger("logs", name="crater_unet_dice", log_graph=True)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
)

trainer.fit(
    unet_dice,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
visualize(unet_dice)

# %%
deeplab_dice = CraterModel(
    arch="DeepLabV3",
    encoder_name="resnet34",
    in_channels=3,
    out_classes=1,
    loss_fn=smp.losses.DiceLoss(
        mode=smp.losses.constants.BINARY_MODE, from_logits=True
    ),
    lr=2e-4,
)

logger = TensorBoardLogger("logs", name="crater_deeplab_dice", log_graph=True)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
)

trainer.fit(
    deeplab_dice,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
visualize(deeplab_dice)

# %%
fpn_focal = CraterModel(
    arch="FPN",
    encoder_name="resnet34",
    in_channels=3,
    out_classes=1,
    loss_fn=smp.losses.FocalLoss(mode=smp.losses.constants.BINARY_MODE),
    lr=2e-4,
)

logger = TensorBoardLogger("logs", name="crater_fpn_focal", log_graph=True)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
)

trainer.fit(
    fpn_focal,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
visualize(fpn_focal)

# %%
unet_focal = CraterModel(
    arch="Unet",
    encoder_name="resnet34",
    in_channels=3,
    out_classes=1,
    loss_fn=smp.losses.FocalLoss(mode=smp.losses.constants.BINARY_MODE),
    lr=2e-4,
)

logger = TensorBoardLogger("logs", name="crater_unet_focal", log_graph=True)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
)

trainer.fit(
    unet_focal,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
visualize(unet_focal)

# %%
deeplab_focal = CraterModel(
    arch="DeepLabV3",
    encoder_name="resnet34",
    in_channels=3,
    out_classes=1,
    loss_fn=smp.losses.FocalLoss(mode=smp.losses.constants.BINARY_MODE),
    lr=2e-4,
)

logger = TensorBoardLogger("logs", name="crater_deeplab_focal", log_graph=True)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
)

trainer.fit(
    deeplab_focal,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
visualize(deeplab_focal)
