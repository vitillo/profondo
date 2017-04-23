import argparse
import logging
import tables
import pandas as pd
import numpy as np
import keras
import os

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, f1_score

OUTPUT_FILE = "predictions.npz"
MODEL_DIRECTORY = "models"
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

    binarizer = MultiLabelBinarizer().fit(meta["genres"])
    genres = binarizer.transform(meta["genres"])
    for genre, p in zip(binarizer.classes_, genres.mean(axis=0)):
        logging.info("{} genre proportion is {:.2f}".format(genre, p))

    # Center images by channel
    for i in range(3):
        images[..., i] -= images[..., i].mean()

    return genres, images, list(binarizer.classes_)


def train_test_split(genres, images, train_ratio=0.7, validation_ratio=0.15):
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

    y_train = genres[train_idx]
    y_test = genres[test_idx]
    y_validation = genres[validation_idx]

    logging.info("Train set size is {}".format(len(x_train)))
    logging.info("Test set size is {}".format(len(x_test)))
    logging.info("Validation set size is {}".format(len(x_validation)))
    return x_train, y_train, x_test, y_test, x_validation, y_validation, test_idx


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


def best_threshold(model, x_val, y_val):
    thresholds = np.arange(0.1, 1, 0.01)
    prediction_proba = model.predict_proba(x_val)
    best_score = -1
    best_threshold = -1

    for t in thresholds:
        prediction = (prediction_proba > t).astype(int)
        score = f1_score(y_val, prediction)
        if score > best_score:
            best_score = score
            best_threshold = t

    assert best_threshold != -1
    return 0.5 # best_threshold TODO: (causes more trouble than anything...)


def predict(model, threshold, x):
    prediction_proba = model.predict_proba(x)
    return (prediction_proba > threshold).astype(int)


def report_callback(epoch, model, x_val, y_val):
    if epoch % 5 == 0:
        threshold = best_threshold(model, x_val, y_val)
        prediction = predict(model, threshold, x_val)
        logging.info("Classification report for validation set:")
        logging.info("\n{}\n".format(classification_report(y_pred=prediction, y_true=y_val)))
        logging.info("Hamming loss:\t\t{:.3f}".format(hamming_loss(y_val, prediction)))
        logging.info("F1 Score:\t\t{:.3f}".format(f1_score(y_val, prediction)))


def filter_genre(x, y):
    is_genre = y == 1
    other_genres = ~is_genre
    length = min(is_genre.sum(), other_genres.sum())
    genre_x = x[is_genre][:length]
    genre_y = y[is_genre][:length]
    other_genres_x = x[other_genres][:length]
    other_genres_y = y[other_genres][:length]
    x = np.concatenate((genre_x, other_genres_x))
    y = np.concatenate((genre_y, other_genres_y))
    idx = np.random.permutation(np.arange(len(x)))
    return x[idx], y[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TMDB Data Exporter")
    parser.add_argument("--logging", help="Enable logging", default=False, action="store_true")
    parser.add_argument("--input-dir", help="Directory with movie data", type=str, default=".")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    args = parser.parse_args()

    if args.logging:
        logging.getLogger().setLevel(logging.INFO)

    genres, images, classes = load_data(args.input_dir)
    x_train, y_train, x_test, y_test, x_val, y_val, test_idx = \
        train_test_split(genres, images)

    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    predictions = np.array([])
    for idx, genre in enumerate(classes):
        logging.info("Training genre model for genre {}".format(genre))

        # Balance training and validation set so that there are as many movies
        # of the selected genre as there are movies that don't belong to it.
        x_train_genre, y_train_genre = filter_genre(x_train, y_train[:, idx])
        model_filename = os.path.join(MODEL_DIRECTORY, "{}_{}".format(genre, MODEL_FILENAME))

        callbacks = [
            ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto'),
            LambdaCallback(on_epoch_end=lambda e, l: report_callback(e, model, x_val, y_val[:, idx]))
        ]

        train_datagen = ImageDataGenerator(horizontal_flip=True)
        train_generator = train_datagen.flow(x_train_genre, y_train_genre, batch_size=args.batch_size)

        model = load_model(model_filename, x_train_genre, y_train_genre)
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(x_train_genre) // args.batch_size,
            epochs=args.epochs,
            verbose=args.logging,
            validation_data=(x_val, y_val[:, idx]),
            callbacks=callbacks)

        # Load best model and evaluate it on the test data
        model = load_model(model_filename, x_train_genre, y_train_genre)
        threshold = best_threshold(model, x_val, y_val[:, idx])
        prediction = predict(model, threshold, x_test)
        if predictions.shape[0] > 0:
            predictions = np.hstack((predictions, prediction))
        else:
            predictions = prediction
        np.savez(OUTPUT_FILE, predictions=predictions, test_idx=test_idx)
