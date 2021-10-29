from __future__ import print_function
from utils import *


def main(args, idx_rep):
    # set session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # initialize experiment
    dataset, network = args.dataset, args.network
    nValid, nQuery, nInit, nStep, nPool = args.nValid, args.nQuery, args.nInit, args.nStep, args.nPool
    path = f'./results/{dataset}_{network}'     # path for saving results
    os.makedirs(path, exist_ok=True)
    print(f'{dataset}/{network}: rep-{idx_rep:02d}')
    (X_train_raw, y_train_raw), (X_test, y_test), n_classes = get_dataset(dataset)
    idx_labeled, idx_unlabeled, idx_valid = set_index(nInit, nValid, y_train_raw, idx_rep)
    valid_set = (X_train_raw[idx_valid], y_train_raw[idx_valid])

    # acquisition step
    test_accs, sigma = [], 0.01
    for step in range(nStep+1):
        # make, train, and test model
        X_labeled, y_labeled = X_train_raw[idx_labeled], y_train_raw[idx_labeled]
        model, acc = train_and_test_model(step, path, args, X_labeled, y_labeled, valid_set, X_test, y_test, n_classes)
        test_accs.append(acc)
        np.savetxt(f'{path}/test_accs_{idx_rep+1:03d}.txt', test_accs, fmt='%.5f')

        # query samples
        if step < nStep:
            np.random.shuffle(idx_unlabeled)
            idx_pool = idx_unlabeled[:nPool]
            X_pool = X_train_raw[idx_pool]
            idx_query, sigma = query_by_DR(model, X_pool, nQuery, X_labeled, y_labeled, sigma)
            idx_labeled = np.concatenate((idx_labeled, idx_pool[idx_query]))
            idx_unlabeled = np.setdiff1d(idx_unlabeled, idx_labeled)

        del model
        tf.compat.v1.keras.backend.clear_session()
