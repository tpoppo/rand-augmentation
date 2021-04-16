import tensorflow as tf
import tensorflow_datasets as tfds


AUTOTUNE = tf.data.experimental.AUTOTUNE


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def dataset_preprocess(dataset, batch_size=256,
                       shuffle=True, before_batch=None):
    dataset = dataset.map(
        normalize_img, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(4 * batch_size)
    if before_batch is not None:
        dataset = dataset.map(before_batch, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_dataset_mnist(batch_size=512,
                      before_batch_train=None,
                      before_batch_test=None):
    ds_train, ds_test = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    ds_train = dataset_preprocess(ds_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  before_batch=before_batch_train)
    ds_test = dataset_preprocess(ds_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 before_batch=before_batch_test)
    return ds_train, ds_test


def get_dataset_cifar100(batch_size=128,
                         before_batch_train=None,
                         before_batch_test=None):
    ds_train, ds_test = tfds.load(
        'cifar100',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    ds_train = dataset_preprocess(ds_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  before_batch=before_batch_train)
    ds_test = dataset_preprocess(ds_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 before_batch=before_batch_test)
    return ds_train, ds_test
