"""
TensorFlow port of train.py — Decoder-only Transformer for Keyword Spotting
Dataset: Google Speech Commands v2 (35 keywords)

Custom train loop (mirrors the PyTorch script step-for-step: OneCycle LR,
grad-norm clipping at 1.0, label-smoothed CE, best-val-acc checkpointing)
rather than model.fit, so behavior is easy to compare against the original.
"""

import json
import math
import os

import tensorflow as tf
from tqdm import tqdm

# Force CPU: tensorflow-metal (Apple's Metal GPU PluggableDevice) has a
# confirmed gradient-correctness bug for this model. Verified directly by
# comparing identical gradient computations: CPU gives a sane global grad
# norm (~2.5) on a freshly initialized model, Metal gives ~1e5-1e6 with
# individual weight-gradient norms in the billions -- values that violate
# basic chain-rule bounds relative to the (small, correct) activation
# gradients feeding them. That corruption is what eventually produces the
# NaN loss/0.0 accuracy after a few epochs. Training is ~2.9x slower on CPU,
# but correct. Revisit this once Apple ships a tensorflow-metal fix. Must run
# before kws_common (or anything else) touches TensorFlow's device machinery.
tf.config.set_visible_devices([], "GPU")

import kws_common as kws

EPOCHS = kws.EPOCHS


# ── LR schedule: PyTorch OneCycleLR (cos anneal_strategy) equivalent ───────────

class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = float(total_steps)
        self.warmup_steps = pct_start * self.total_steps
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        def warmup():
            pct = step / tf.maximum(self.warmup_steps, 1.0)
            cos_out = (1.0 - tf.cos(math.pi * pct)) / 2.0
            return self.initial_lr + (self.max_lr - self.initial_lr) * cos_out

        def anneal():
            pct = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
            cos_out = (1.0 + tf.cos(math.pi * pct)) / 2.0
            return self.min_lr + (self.max_lr - self.min_lr) * cos_out

        return tf.cond(step < self.warmup_steps, warmup, anneal)

    def get_config(self):
        return {"max_lr": self.max_lr, "total_steps": self.total_steps}


def train():
    os.makedirs(kws.CKPT_DIR, exist_ok=True)
    os.makedirs(kws.DATA_ROOT, exist_ok=True)

    print("Loading datasets (downloads ~2.4 GB on first run)...")
    train_pairs = kws.list_split(kws.DATA_ROOT, "training")
    val_pairs = kws.list_split(kws.DATA_ROOT, "validation")
    print(f"  Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}")

    train_ds = kws.make_dataset(train_pairs, kws.BATCH_SIZE, shuffle=True, augment=True)
    val_ds = kws.make_dataset(val_pairs, kws.BATCH_SIZE, shuffle=False, augment=False)
    steps_per_epoch = math.ceil(len(train_pairs) / kws.BATCH_SIZE)

    # GPU is force-disabled above (tensorflow-metal gradient bug). Note
    # list_physical_devices() still lists it (it enumerates hardware,
    # ignoring the visibility mask) -- get_visible_devices() is the one that
    # reflects what set_visible_devices() actually did.
    device = "GPU" if tf.config.get_visible_devices("GPU") else "CPU"

    model = kws.build_model()
    model.summary()
    n_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"Parameters: {n_params:,}  |  Device: {device}")

    lr_schedule = OneCycleLR(kws.LR, total_steps=steps_per_epoch * EPOCHS)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=kws.WEIGHT_DECAY, beta_1=0.9, beta_2=0.98,
    )
    # Keras's SparseCategoricalCrossentropy has no label_smoothing argument (only
    # the one-hot CategoricalCrossentropy does), so we one-hot the integer labels
    # ourselves to get the same label-smoothed loss as PyTorch's
    # nn.CrossEntropyLoss(label_smoothing=0.1).
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

    @tf.function
    def train_step(mel, labels):
        with tf.GradientTape() as tape:
            logits = model(mel, training=True)
            labels_oh = tf.one_hot(labels, kws.NUM_CLASSES)
            loss = loss_fn(labels_oh, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.float32))
        return loss, correct

    @tf.function
    def val_step(mel, labels):
        logits = model(mel, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(labels, tf.int32)), tf.float32))
        return correct

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        total_loss = total_correct = total = 0.0
        train_bar = tqdm(train_ds, total=steps_per_epoch, desc=f"Epoch {epoch:3d}/{EPOCHS} [train]", leave=False)
        for mel, labels in train_bar:
            bsz = float(labels.shape[0] or mel.shape[0])
            loss, correct = train_step(mel, labels)
            total_loss += float(loss) * bsz
            total_correct += float(correct)
            total += bsz
            train_bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{total_correct/total:.4f}")

        train_acc = total_correct / total

        val_correct = val_total = 0.0
        val_bar = tqdm(val_ds, desc=f"Epoch {epoch:3d}/{EPOCHS} [val]  ", leave=False)
        for mel, labels in val_bar:
            bsz = float(labels.shape[0] or mel.shape[0])
            correct = val_step(mel, labels)
            val_correct += float(correct)
            val_total += bsz
            val_bar.set_postfix(acc=f"{val_correct/val_total:.4f}")
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"loss {total_loss/total:.4f} | "
            f"train {train_acc:.4f} | val {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(kws.WEIGHTS_PATH)
            config = dict(
                n_mels=kws.N_MELS, d_model=kws.D_MODEL, n_heads=kws.N_HEADS,
                n_layers=kws.N_LAYERS, d_ff=kws.D_FF, dropout=0.0,
                num_classes=kws.NUM_CLASSES, max_len=kws.MAX_FRAMES,
                epoch=epoch, val_acc=val_acc,
            )
            with open(kws.CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  ✓ Saved best model  val_acc={val_acc:.4f}")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
