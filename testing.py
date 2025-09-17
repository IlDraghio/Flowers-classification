import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os, re, random

RES_LIST = ["192x192","224x224","331x331","512x512"]

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
    image = tf.cast(image, tf.float32)

    image = preprocess_input(image)
    
    if with_label:
        label = features["class"]
        return image, label
    else:
        return image, features["id"]


# -------- Dataset loader --------
def make_ds(tfrecord_dir, target_size=(224,224), batch_size=32, with_label=True):
    files = tf.io.gfile.glob(tfrecord_dir + "/*.tfrec")
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {tfrecord_dir}")

    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(
        lambda x: parse_example(x, target_size=target_size, with_label=with_label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds


# -------- Show one image from a dataset --------
def show_one_from_ds(ds, name, count):
    rand = random.randrange(count)
    try:
        example = ds.unbatch().skip(rand).take(1)
        for img, lbl in example:
            plt.figure()
            plt.imshow(img.numpy())
            plt.title(f"{name} - Label/ID: {lbl.numpy()}")
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
    train_count,val_count,test_count = int(0),int(0),int(0)
    for res in RES_LIST:
        train_count += count_from_filenames(f"/kaggle/input/dataset-flowers/dataset/{res}/train")
        val_count   += count_from_filenames(f"/kaggle/input/dataset-flowers/dataset/{res}/val")
        test_count   += count_from_filenames(f"/kaggle/input/dataset-flowers/dataset/{res}/val")
    
    batch_size = 32

    train = [make_ds(
        f"/kaggle/input/dataset-flowers/dataset/{res}/train",
        target_size=(224,224),
        batch_size=batch_size,
        with_label=True
    ) for res in RES_LIST]
    
    train_ds = train[0]
    for ds in train[1:]:
        train_ds = train_ds.concatenate(ds)
    
    train_ds = train_ds.shuffle(10000).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    val = [make_ds(
        f"/kaggle/input/dataset-flowers/dataset/{res}/val",
        target_size=(224,224),
        batch_size=batch_size,
        with_label=True
    ) for res in RES_LIST]
    
    val_ds = val[0]
    for ds in val[1:]:
        val_ds = val_ds.concatenate(ds)

    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test = [make_ds(
        f"/kaggle/input/dataset-flowers/dataset/{res}/test",
        target_size=(224,224),
        batch_size=batch_size,
        with_label=True
    ) for res in RES_LIST]
    
    test_ds = val[0]
    for ds in val[1:]:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    #show_one_from_ds(train_ds, "Train", train_count)
    #show_one_from_ds(val_ds, "Val", val_count)
    #show_one_from_ds(test_ds, "Test", test_count)

    print("Train size:", train_count)
    print("Val size:", val_count)
    print("Test size:", test_count)
    
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

batch_size = 32
steps_per_epoch = train_count // batch_size
validation_steps = val_count // batch_size

NUM_CLASSES = 104  # <-- change to match your dataset

with strategy.scope():
    base_model = tf.keras.applications.EfficientNetV2B3(
    input_shape=(224, 224, 3),   # EfficientNetV2B3 default
    include_top=False,
    weights="imagenet"
    )

    base_model.trainable = False  # freeze backbone for transfer learning

    inputs = tf.keras.Input(shape=(224,224,3))
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

early_stop = EarlyStopping(
    monitor="val_loss",       # watch validation loss
    patience=3,               # stop after 3 epochs with no improvement
    restore_best_weights=True # roll back to best model
)

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
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    callbacks=[early_stop]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
    ModelCheckpoint("best_finetuned.h5", monitor="val_loss", save_best_only=True)
]

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
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks
)

import matplotlib.pyplot as plt

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