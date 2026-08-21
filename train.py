"""
Decoder-only Transformer for Keyword Spotting
Dataset: Google Speech Commands v2 (35 keywords)

Same training loop as the original monolithic script; dataset/model classes
now live in kws_common_torch.py so convert_to_onnx.py and run_onnx_mac.py
can reuse them without drift.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import kws_common_torch as kws
from kws_common_torch import (
    SpeechCommandsDataset, collate_fn, KeywordSpottingTransformer,
)


def train():
    os.makedirs(kws.CKPT_DIR, exist_ok=True)
    os.makedirs(kws.DATA_ROOT, exist_ok=True)

    print("Loading datasets (downloads ~2.4 GB on first run)...")
    train_ds = SpeechCommandsDataset(kws.DATA_ROOT, "training",   augment=True)
    val_ds   = SpeechCommandsDataset(kws.DATA_ROOT, "validation", augment=False)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # pin_memory only helps (and is only supported) for CUDA host->device
    # copies; it's a no-op/irrelevant for MPS and CPU.
    pin_memory = kws.DEVICE == "cuda"
    train_dl = DataLoader(train_ds, kws.BATCH_SIZE, shuffle=True,
                          num_workers=2, collate_fn=collate_fn, pin_memory=pin_memory)
    val_dl   = DataLoader(val_ds, kws.BATCH_SIZE, shuffle=False,
                          num_workers=2, collate_fn=collate_fn, pin_memory=pin_memory)

    model = KeywordSpottingTransformer().to(kws.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  |  Device: {kws.DEVICE}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=kws.LR, weight_decay=kws.WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=kws.LR,
        steps_per_epoch=len(train_dl), epochs=kws.EPOCHS,
    )

    best_val_acc = 0.0
    for epoch in range(1, kws.EPOCHS + 1):
        model.train()
        total_loss = total_correct = total = 0
        train_bar = tqdm(train_dl, desc=f"Epoch {epoch:3d}/{kws.EPOCHS} [train]", leave=False)
        for mel, labels in train_bar:
            mel, labels = mel.to(kws.DEVICE), labels.to(kws.DEVICE)
            logits = model(mel)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss    += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += labels.size(0)
            train_bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{total_correct/total:.4f}")

        train_acc = total_correct / total

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            val_bar = tqdm(val_dl, desc=f"Epoch {epoch:3d}/{kws.EPOCHS} [val]  ", leave=False)
            for mel, labels in val_bar:
                mel, labels = mel.to(kws.DEVICE), labels.to(kws.DEVICE)
                preds        = model(mel).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                val_bar.set_postfix(acc=f"{val_correct/val_total:.4f}")
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:3d}/{kws.EPOCHS} | "
            f"loss {total_loss/total:.4f} | "
            f"train {train_acc:.4f} | val {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "config": dict(
                        n_mels=kws.N_MELS, d_model=kws.D_MODEL, n_heads=kws.N_HEADS,
                        n_layers=kws.N_LAYERS, d_ff=kws.D_FF, dropout=0.0,
                        num_classes=kws.NUM_CLASSES, max_len=kws.MAX_FRAMES,
                    ),
                },
                kws.CKPT_PATH,
            )
            print(f"  ✓ Saved best model  val_acc={val_acc:.4f}")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
