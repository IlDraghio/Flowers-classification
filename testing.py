import tensorflow as tf
import matplotlib.pyplot as plt
import os, re

# -------- TFRecord parsing --------
def parse_example(example_proto, target_size=(224,224), with_label=True):
    feature_description = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    if with_label:
        feature_description["class"] = tf.io.FixedLenFeature([], tf.int64)

    features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode + preprocess image
    image = tf.io.decode_jpeg(features["image"], channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0

    if with_label:
        label = features["class"]
        return image, label
    else:
        return image, features["id"]


# -------- Dataset loader --------
def make_ds(tfrecord_dir, target_size=(224,224), batch_size=32, with_label=True, repeat=False):
    files = tf.io.gfile.glob(tfrecord_dir + "/*.tfrec")
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {tfrecord_dir}")

    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(
        lambda x: parse_example(x, target_size=target_size, with_label=with_label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -------- Show one image from a dataset --------
def show_one_from_ds(ds, name):
    try:
        images, labels_or_ids = next(iter(ds))
        img = images[0].numpy()
        lbl = labels_or_ids[0].numpy()

        plt.figure()
        plt.imshow(img)
        plt.title(f"{name} - Label/ID: {lbl}")
        plt.axis("off")
        plt.show()
    except StopIteration:
        print(f"{name}: dataset empty")

def count_from_filenames(path):
    total = 0
    for fname in os.listdir(path):
        m = re.search(r"-(\d+)\.tfrec$", fname)
        if m:
            total += int(m.group(1))
    return total

# -------- Example usage --------
if __name__ == "__main__":
    train_count = count_from_filenames("/kaggle/input/dataset-flowers/dataset/192x192/train")
    val_count   = count_from_filenames("/kaggle/input/dataset-flowers/dataset/192x192/val")
    
    batch_size = 32
    
    train_ds = make_ds(
        "/kaggle/input/dataset-flowers/dataset/192x192/train",
        target_size=(192,192),
        batch_size=batch_size,
        with_label=True,
        repeat=True   # ✅ important for TPU
    )
    
    val_ds = make_ds(
        "/kaggle/input/dataset-flowers/dataset/192x192/val",
        target_size=(192,192),
        batch_size=batch_size,
        with_label=True,
        repeat=True   # ✅ same here
    )
    
    test_ds = make_ds(
        "/kaggle/input/dataset-flowers/dataset/192x192/test",
        target_size=(192,192),
        batch_size=batch_size,
        with_label=False,
        repeat=False
    )

    show_one_from_ds(train_ds, "Train")
    show_one_from_ds(val_ds, "Val")
    show_one_from_ds(test_ds, "Test")

    print("Train size:", count_from_filenames("/kaggle/input/dataset-flowers/dataset/192x192/train"))
    print("Val size:", count_from_filenames("/kaggle/input/dataset-flowers/dataset/192x192/val"))
    print("Test size:", count_from_filenames("/kaggle/input/dataset-flowers/dataset/192x192/test"))
    

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    print("✅ TPU detected:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    tpu = None
    print("❌ No TPU found")
    strategy = tf.distribute.get_strategy()

print("✅ Using strategy:", strategy)

# List devices
print("Devices:", tf.config.list_logical_devices())
print("Logical devices:", tf.config.list_logical_devices())

NUM_CLASSES = 104  # <-- change to match your dataset

steps_per_epoch = train_count // batch_size
validation_steps = val_count // batch_size

print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)

with strategy.scope():
    base_model = tf.keras.applications.ResNet152V2(
        input_shape=(192,192,3),   # adjust to your dataset image size
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # freeze backbone for transfer learning

    inputs = tf.keras.Input(shape=(192,192,3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

with strategy.scope():
    # Freeze backbone (ResNet)
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

with strategy.scope():
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # smaller LR
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

def plot_history(h1, h2):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.legend()
    plt.show()

plot_history(history_head, history_finetune)