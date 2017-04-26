import argparse
import logging
import tables
import pandas as pd
import numpy as np
import keras

from keras import backend as K
from keras import applications as kapps
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from os.path import join, exists
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_similarity_score

MODEL_FILENAME = "model.h5"
TOP_MODEL_WEIGHTS_FILE = "top_model_weights.h5"
IMAGE_FILENAME = "image_export.h5"
METADATA_FILENAME = "metadata_export"
TB_DIR = 'tensorboard'


def load_data(path, sample_limit):
    images = tables.open_file(join(path, IMAGE_FILENAME), "r").get_node("/images")[:]
    meta = pd.read_pickle(join(path, METADATA_FILENAME))
    meta.index = np.arange(len(meta))

    assert(len(meta["tmdb_id"].unique()) == len(meta))
    assert(len(images) == len(meta))
    assert K.image_data_format() == 'channels_last'

    # Center images by channel
    for i in range(3):
        images[..., i] -= images[..., i].mean()

    sample_limit = sample_limit if sample_limit else len(meta)
    return meta.iloc[:sample_limit], images[:sample_limit]


def compute_genre_weights(y_train):
    # See sklearn.utils.class_weight.compute_class_weight
    genre_weights = y_train.sum()/(y_train.shape[1]*y_train.sum(axis=0).astype(float))
    return dict(enumerate(genre_weights))


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


def build_pre_trained(network_name, x_train, y_train, x_val, y_val, epochs, batch_size,
                      num_to_unfreeze, use_tensorboard):
    supported_models = {
        'vgg16': {
            'class': kapps.VGG16,
            'params': {
                'weights': 'imagenet',
                'include_top': False,
                'input_shape': x_train.shape[1:]
            }
        }
    }

    if not network_name in supported_models:
        raise Exception("Unsupported pre-trained model {}".format(network_name))

    # Load the convolutional part from a pre-trained model.
    model_params = supported_models[network_name]['params']
    model = supported_models[network_name]['class'](**model_params)

    # Build a classifier model to put on top of the convolutional model.
    def generate_top_model(input_shape, output_shape):
        m = Sequential()
        m.add(Flatten(input_shape=input_shape))
        m.add(Dense(4096, activation='relu'))
        m.add(Dense(4096, activation='relu'))
        m.add(Dropout(0.5))
        m.add(Dense(1024, activation='relu'))
        m.add(Dropout(0.25))
        m.add(Dense(output_shape, activation='sigmoid'))
        return m

    if not exists(TOP_MODEL_WEIGHTS_FILE):
        # Build the "bottleneck-features" to feed into the top layers, to init
        # the output layers with reasonable features: we basically record the
        # output of the last convolutional block with our data and feed that
        # into the new top block.
        # Note: our images are already scaled in the [0, 1] range, we don't need
        # the "rescale" part as required by the pre-trained networks. We
        # don't shuffle the samples to be able to pick their labels later on.
        bottleneck_train = model.predict(x_train, batch_size, verbose=True)
        bottleneck_val = model.predict(x_val, batch_size, verbose=True)

        genre_weights = compute_genre_weights(y_train)

        # Compile and train the top_model with the "bottleneck-features".
        top_model = generate_top_model(bottleneck_train.shape[1:], y_train.shape[1])
        top_model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9))
        logging.info(top_model.summary())
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=10e-6),
            ModelCheckpoint(filepath=TOP_MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True,
                            save_weights_only=True, verbose=1)
        ]

        # Optionally enable tensorboard.
        if use_tensorboard:
            callbacks.append(TensorBoard(log_dir=join(TB_DIR, 'top'), histogram_freq=1,
                                         write_graph=True, write_images=False))

        top_model.fit(bottleneck_train, y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(bottleneck_val, y_val),
                      class_weight=genre_weights,
                      callbacks=callbacks,
                      verbose=True)

    # Instantiate another top_model. We need to do the weights save/load dance
    # as we cannot modify a model after it was compiled, so we cannot use the
    # previous |top_model| again.
    trained_top = generate_top_model(model.output_shape[1:], y_train.shape[1])
    trained_top.load_weights(TOP_MODEL_WEIGHTS_FILE)

    # Set all the convolutional layer in the pre-trained network to
    # to non-trainable (weights will not be updated).
    for layer in model.layers[:-num_to_unfreeze]:
        layer.trainable = False

    # Join the convolutional part and the top layers. See
    # https://github.com/fchollet/keras/issues/4040#issuecomment-253309615
    joint_model = Model(input=model.input, output=trained_top(model.output))

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    joint_model.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9))
    logging.info(joint_model.summary())
    return joint_model


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


def report_callback(epoch, model, x_val, y_val):
    def recall(y_true, y_pred):
        tp = (y_true & y_pred).sum(axis=1)
        p = y_true.sum(axis=1)
        ratio = np.nan_to_num(tp/p.astype(float))
        return ratio.mean()

    def precision(y_true, y_pred):
        tp = (y_pred & y_true).sum(axis=1)
        pp = y_pred.sum(axis=1)
        ratio = np.nan_to_num(tp/pp.astype(float))
        return ratio.mean()

    if epoch % 5 == 0:
        prediction_proba = model.predict(x_val)
        prediction = (prediction_proba > 0.5).astype(int)

        logging.info("Classification report")
        logging.info("\n{}\n".format(classification_report(y_val, prediction)))
        logging.info("Hamming loss:\t\t{:.3f}".format(hamming_loss(y_val, prediction)))
        logging.info("Jaccard similarity:\t{:.3f}".format(jaccard_similarity_score(y_val, prediction)))
        logging.info("Precision:\t{:.3f}".format(precision(y_val, prediction)))
        logging.info("Recall:\t{:.3f}".format(recall(y_val, prediction)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TMDB Data Exporter")
    parser.add_argument("--logging", help="Enable logging", default=False, action="store_true")
    parser.add_argument("--input-dir", help="Directory with movie data", type=str, default=".")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--pre-trained", help="Pre-trained network name", type=str)
    parser.add_argument("--unfreeze-layers", type=int, default=4,
                        help="Number of convolutional layers to unfreeze in the pre-trained network")
    parser.add_argument("--epochs", help="Epochs", type=int, default=100)
    parser.add_argument("--tensorboard", help="Enable tensorboard", default=False, action="store_true")
    parser.add_argument("--sample-limit", help="Maximum number of samples to use", default=None, type=int)
    args = parser.parse_args()

    if args.logging:
        logging.getLogger().setLevel(logging.INFO)

    meta, images = load_data(args.input_dir, args.sample_limit)
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_split(meta, images)

    genre_weights = compute_genre_weights(y_train)
    logging.info("Genre weights: {}".format(genre_weights))

    callbacks = [
        ModelCheckpoint(filepath=MODEL_FILENAME, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto'),
        LambdaCallback(on_epoch_end=lambda e, l: report_callback(e, model, x_val, y_val))
    ]

    if args.tensorboard:
        callbacks.append(TensorBoard(log_dir=join(TB_DIR, 'full'), histogram_freq=1,
                                     write_graph=True, write_images=False))

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(x_train, y_train, batch_size=args.batch_size)

    model = build_pre_trained(args.pre_trained, x_train, y_train, x_val, y_val, args.epochs,
                              args.batch_size, args.unfreeze_layers, args.tensorboard)\
            if args.pre_trained else load_model(x_train, y_train)
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train) // args.batch_size,
        epochs=args.epochs,
        verbose=args.logging,
        validation_data=(x_val, y_val),
        class_weight=genre_weights,
        callbacks=callbacks)

    # Load best model and evaluate it on the test data
    assert exists(MODEL_FILENAME)
    model = load_model(x_test, y_test)
    report_callback(0, model, x_test, y_test)
