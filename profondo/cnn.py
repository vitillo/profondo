import argparse
import logging
import tables
import pandas as pd
import numpy as np
import keras

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss

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


def train_test_split(genre, meta, images, train_ratio=0.7, validation_ratio=0.15):
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

    genres = np.array(meta["genres"].apply(lambda x: genre in x)).astype(int)
    y_train = genres[train_idx]
    y_test = genres[test_idx]
    y_validation = genres[validation_idx]

    logging.info("Train set size is {}".format(len(x_train)))
    logging.info("Test set size is {}".format(len(x_test)))
    logging.info("Validation set size is {}".format(len(x_validation)))
    return x_train, y_train, x_test, y_test, x_validation, y_validation


def build_model(x_train, y_train):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=x_train.shape[1:]))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def load_model(model_filename, x_train, y_train):
    try:
        model = keras.models.load_model(model_filename)
        logging.info("Checkpointed model loaded")
    except Exception:
        model = build_model(x_train, y_train)
        logging.info("No checkpointed model found - building a new one")

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])

    logging.info(model.summary())
    return model


def report_callback(epoch, model, x_val, y_val):
    if epoch % 5 == 0:
        prediction_proba = model.predict_proba(x_val)
        prediction = (prediction_proba > 0.5).astype(int)

        logging.info("Classification report for validation set:")
        logging.info("\n{}\n".format(classification_report(y_pred=prediction, y_true=y_val)))
        logging.info("Hamming loss:\t\t{:.3f}".format(hamming_loss(y_val, prediction)))


def filter_data(meta, images, genre):
    is_genre = np.array(meta["genres"].apply(lambda x: genre in x))
    other_genres = ~is_genre
    length = min(is_genre.sum(), other_genres.sum())

    genre_meta = meta[is_genre].iloc[:length]
    other_genres_meta = meta[other_genres].iloc[:length]
    meta = genre_meta.append(other_genres_meta)
    meta.index = range(len(meta))

    genre_images = images[is_genre][:length]
    other_genres_images = images[other_genres][:length]
    images = np.concatenate((genre_images, other_genres_images))

    assert len(meta) == len(images)
    return meta, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TMDB Data Exporter")
    parser.add_argument("--logging", help="Enable logging", default=False, action="store_true")
    parser.add_argument("--input-dir", help="Directory with movie data", type=str, default=".")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    parser.add_argument("--genre", help="Genre to train the model on", type=str, default="Drama")
    args = parser.parse_args()

    if args.logging:
        logging.getLogger().setLevel(logging.INFO)

    meta, images = load_data(args.input_dir)
    meta, images = filter_data(meta, images, args.genre)
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_split(args.genre, meta, images)
    model_filename = "{}_{}".format(args.genre.lower(), MODEL_FILENAME)

    callbacks = [
        ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto'),
        LambdaCallback(on_epoch_end=lambda e, l: report_callback(e, model, x_val, y_val))
    ]

    train_datagen = ImageDataGenerator(
        horizontal_flip=True
    )
    train_generator = train_datagen.flow(x_train, y_train, batch_size=args.batch_size)

    model = load_model(model_filename, x_train, y_train)
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train) // args.batch_size,
        epochs=args.epochs,
        verbose=args.logging,
        validation_data=(x_val, y_val),
        callbacks=callbacks)
