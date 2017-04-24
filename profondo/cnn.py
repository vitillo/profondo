import argparse
import logging
import tables
import pandas as pd
import numpy as np
import keras

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_similarity_score

MODEL_FILENAME = "model.h5"
IMAGE_FILENAME = "image_export.h5"
METADATA_FILENAME = "metadata_export"


def load_data(path):
    images = tables.open_file(join(path, IMAGE_FILENAME), "r").get_node("/images")[:]
    meta = pd.read_pickle(join(path, METADATA_FILENAME))
    meta.index = np.arange(len(meta))

    assert(len(meta["tmdb_id"].unique()) == len(meta))
    assert(len(images) == len(meta))
    assert K.image_data_format() == 'channels_last'

    return meta, images


def train_test_split(meta, images, train_ratio=0.7, validation_ratio=0.15):
    np.random.seed(42)
    idx = np.random.permutation(np.arange(len(images)))
    train_size = int(len(idx)*train_ratio)
    validation_size = int(len(idx)*validation_ratio)

    train_idx_end = train_size
    validation_idx_end = train_size + validation_size

    train_idx = idx[:train_idx_end]
    validation_idx = idx[train_idx_end:validation_idx_end]
    test_idx = idx[validation_idx_end:]

    x_train = images[train_idx]
    x_test = images[test_idx]
    x_validation = images[validation_idx]

    binarizer = MultiLabelBinarizer().fit(meta["genres"])
    genres = binarizer.transform(meta["genres"])
    y_train = genres[train_idx]
    y_test = genres[test_idx]
    y_validation = genres[validation_idx]

    for genre, p in zip(binarizer.classes_, genres.mean(axis=0)):
        logging.info("{} genre proportion is {:.2f}".format(genre, p))

    logging.info("Train set size is {}".format(len(x_train)))
    logging.info("Test set size is {}".format(len(x_test)))
    logging.info("Validation set size is {}".format(len(x_validation)))
    return x_train, y_train, x_test, y_test, x_validation, y_validation


def build_model(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    return model


def load_model(x_train, y_train):
    try:
        model = keras.models.load_model(MODEL_FILENAME)
        logging.info("Checkpointed model loaded")
    except Exception:
        model = build_model(x_train, y_train)
        logging.info("No checkpointed model found - building a new one")

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam())

    logging.info(model.summary())
    return model


def report_callback(epoch, x_val, y_val):
    if epoch % 5 == 0:
        prediction_proba = model.predict_proba(x_val)
        prediction = (prediction_proba > 0.5).astype(int)

        logging.info("Classification report for validation set:")
        logging.info("\n{}\n".format(classification_report(y_pred=prediction, y_true=y_val)))
        logging.info("Hamming loss:\t\t{:.3f}".format(hamming_loss(y_val, prediction)))
        logging.info("Jaccard similarity:\t{:.3f}".format(jaccard_similarity_score(y_val, prediction)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TMDB Data Exporter")
    parser.add_argument("--logging", help="Enable logging", default=False, action="store_true")
    parser.add_argument("--input-dir", help="Directory with movie data", type=str, default=".")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    args = parser.parse_args()

    if args.logging:
        logging.getLogger().setLevel(logging.INFO)

    meta, images = load_data(args.input_dir)
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_split(meta, images)
    genre_weights = (1 / y_train.mean(axis=0))
    genre_weights = dict(enumerate(genre_weights/genre_weights.max()))

    callbacks = [
        ModelCheckpoint(filepath=MODEL_FILENAME, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=10, verbose=1, min_lr=10e-6),
        LambdaCallback(on_epoch_end=lambda e, l: report_callback(e, x_val, y_val))
    ]

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(x_train, y_train, batch_size=args.batch_size)
    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow(x_val, y_val, batch_size=args.batch_size)

    model = load_model(x_train, y_train)
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train) // args.batch_size,
        epochs=args.epochs,
        verbose=args.logging,
        validation_data=validation_generator,
        validation_steps=len(x_val) // args.batch_size,
        class_weight=genre_weights,
        callbacks=callbacks)
