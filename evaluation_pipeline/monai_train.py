import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import monai
from monai.data import Dataset, decollate_batch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped,
    AsDiscreted
)

def main():
    parser = argparse.ArgumentParser(description="MONAI Training Script for Segmentation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, choices=["SwinUNETR", "UNet"], default="SwinUNETR")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images_tr = sorted([os.path.join(args.data_dir, 'imagesTr', f) for f in os.listdir(os.path.join(args.data_dir, 'imagesTr'))])
    labels_tr = sorted([os.path.join(args.data_dir, 'labelsTr', f) for f in os.listdir(os.path.join(args.data_dir, 'labelsTr'))])
    
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images_tr, labels_tr)]
    val_split = int(len(data_dicts) * 0.2)
    train_files, val_files = data_dicts[:-val_split], data_dicts[-val_split:]

    roi_size = (96, 96, 96)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=roi_size, pos=1, neg=1, num_samples=4),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    if args.model_name == "SwinUNETR":
        model = SwinUNETR(img_size=roi_size, in_channels=1, out_channels=2, feature_size=48, use_checkpoint=True).to(device)
    else:
        model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)).to(device)
        
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_metric = -1
    best_metric_epoch = -1
    # --- THIS IS THE CORRECTED SECTION ---
    # We apply AsDiscrete to the model output ('pred') and the ground truth ('label')
    post_pred = AsDiscreted(keys="pred", argmax=True, to_onehot=2)
    post_label = AsDiscreted(keys="label", to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    for epoch in range(args.max_epochs):
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        model.train()
        epoch_loss = 0
        for batch_data in tqdm(train_loader, desc="Training"):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(train_loader):.4f}")

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validating"):
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    
                    val_outputs = sliding_window_inference(val_inputs, roi_size, 4, model)
                    
                    # Create dictionaries for post-processing
                    batch_list = [{"pred": pred, "label": label} for pred, label in zip(decollate_batch(val_outputs), decollate_batch(val_labels))]
                    
                    processed_outputs = [post_pred(item) for item in batch_list]
                    processed_labels = [post_label(item) for item in batch_list]
                    
                    # Extract the tensors for the metric
                    dice_metric(y_pred=[d["pred"] for d in processed_outputs], y=[d["label"] for d in processed_labels])

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                print(f"Current validation mean dice: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_metric_model.pth"))
                    print("Saved new best model")
    
    print(f"Training complete. Best validation dice: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    main()