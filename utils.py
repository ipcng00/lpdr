import os
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np


def get_dataset(dataset):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3) / 255.0
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3) / 255.0

    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    n_classes = len(np.unique(y_train))

    return (X_train, y_train), (X_test, y_test), n_classes


def set_index(nInit, nValid, y_train_raw, random_seed=0):
    n_classes = len(np.unique(y_train_raw))
    np.random.seed(random_seed)
    idx_shuffled = np.random.permutation(len(y_train_raw))
    idx = [idx_shuffled[np.where(y_train_raw[idx_shuffled] == (i % n_classes))[0][i // n_classes]] for i in range(nInit+nValid)]
    idx_labeled, idx_valid = np.array(idx[:nInit]), np.array(idx[nInit:])
    idx_unlabeled = np.setdiff1d(idx_shuffled, np.concatenate((idx_labeled, idx_valid)))
    np.random.shuffle(idx_labeled)
    np.random.shuffle(idx_unlabeled)
    np.random.shuffle(idx_valid)

    return idx_labeled, idx_unlabeled, idx_valid


def get_model(network, input_shape, n_classes):
    if network.upper() in ['SCNN', 'S-CNN']:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_classes, kernel_initializer='he_normal'),
            keras.layers.Activation('softmax')
        ])
        opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    elif network.upper() in ['KCNN', 'K-CNN']:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_classes, kernel_initializer='he_normal'),
            keras.layers.Activation('softmax')
        ])
        opt = keras.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train_and_test_model(step, path, args, X_labeled, y_labeled, valid_set, X_test, y_test, n_classes):
    network, nBatch, nEpoch, nStep = args.network, args.nBatch, args.nEpoch, args.nStep
    weights_best_file = f'{path}/weights_best.hdf5'
    if os.path.exists(weights_best_file): os.remove(weights_best_file)
    model = get_model(network, X_labeled.shape[1:], n_classes)
    callback = [tf.keras.callbacks.ModelCheckpoint(filepath=weights_best_file, monitor='val_accuracy', save_best_only=True, verbose=1)]
    print(f'Training.. step {step + 1:03d}/{nStep:03d}') if step < nStep else print(f'Training.. final step')
    model.fit(X_labeled, y_labeled, batch_size=nBatch, epochs=nEpoch, validation_data=valid_set, callbacks=callback)
    if os.path.exists(weights_best_file): model.load_weights(weights_best_file)
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f'Step {step + 1:03d}/{nStep:03d} - test acc: {acc:.5f}\n') if step < nStep else print(f'Final step - test acc: {acc:.5f}\n')
    if os.path.exists(weights_best_file): os.remove(weights_best_file)

    return model, acc


def get_feature(model, X, feature=None):
    idx_layer = [l for l in range(len(model.layers)) if len(model.layers[l].get_weights()) > 0][-1]
    model_output = K.function([model.layers[0].input], model.layers[idx_layer - 1].output)
    idx_split = np.linspace(0, X.shape[0], 11).astype(int)
    for i in range(10):
        feature = np.concatenate((feature, model_output(X[idx_split[i]:idx_split[i+1]]))) if feature is not None else model_output(X[idx_split[i]:idx_split[i+1]])
    w_hat = model.layers[idx_layer].get_weights()

    return feature, w_hat


def get_label(feature, weight):
    output = np.matmul(feature, weight[0]) + np.reshape(weight[1], (1, -1))
    label = np.argmax(output, axis=1)

    return label


def query_by_DR(model, X_pool, nQuery, X_labeled, y_labeled, sigma, N=100, beta=1.0):
    rho_target = nQuery / X_pool.shape[0]
    feature_L, w_hat = get_feature(model, X_labeled)
    feature_P, _ = get_feature(model, X_pool)
    y_hat = get_label(feature_P, w_hat)
    e_hat = np.mean(y_labeled != get_label(feature_L, w_hat))
    gammas, ys = [], []
    for n in range(N):
        w_n = [np.random.normal(w, sigma) for w in w_hat]
        e_n = np.mean(y_labeled != get_label(feature_L, w_n))
        gammas.append(np.exp(-(e_n - e_hat)))
        y_n = get_label(feature_P, w_n)
        ys.append(y_n)
        rho_n = np.mean(y_n != y_hat)
        sigma *= np.exp(-beta * (rho_n - rho_target))
    gammas = np.array(gammas).reshape((-1, 1))
    ys = np.array(ys)
    wdr = np.sum(gammas * (ys != y_hat), axis=0) / np.sum(gammas)       # weighted disagree ratio
    idx_query = np.argsort(wdr)[::-1][:nQuery]

    return idx_query, sigma
