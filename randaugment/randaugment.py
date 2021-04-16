import tensorflow as tf
import tensorflow_addons as tfa

_MAX_LEVEL = 30


@tf.function
def aug_contrast(image, m):
    return tf.image.random_contrast(image,
                                    1 - 0.4 * m / _MAX_LEVEL,
                                    1 + 0.4 * m / _MAX_LEVEL)


@tf.function
def aug_invert(image, m):
    return 1 - image


@tf.function
def aug_brightness(image, m):
    return tf.image.random_brightness(image, m / _MAX_LEVEL * 1.8 + 0.1)


@tf.function
def aug_rotate(image, m):
    m = m / _MAX_LEVEL * \
        tf.random.uniform([], minval=-30, maxval=30, dtype=tf.float32)
    return tfa.image.rotate(image, m)


@tf.function
def aug_shear_x(image, m):
    color = [0, 0, 0]
    image = tfa.image.shear_x(image, m / _MAX_LEVEL * 0.3, replace=color)
    return image


@tf.function
def aug_shear_y(image, m):
    color = [0, 0, 0]
    image = tfa.image.shear_y(image, m / _MAX_LEVEL * 0.3, replace=color)
    return image


@tf.function
def aug_translate(image, m):
    return tfa.image.translate(image, image.shape[1] * m / _MAX_LEVEL * tf.random.uniform([2], minval=-0.3, maxval=0.3, dtype=tf.float32))


@tf.function
def aug_blur(image, m):
    return tfa.image.gaussian_filter2d(image, sigma=m / _MAX_LEVEL)


@tf.function
def aug_cutout(image, m):
    m = tf.cast(m / _MAX_LEVEL * 0.7 * image.shape[1], tf.int32)
    m = tf.bitwise.bitwise_and(m, 1073741822)  # 1073741822 = 2^30 - 2
    image = tf.expand_dims(image, 0)
    return tfa.image.random_cutout(image, (m, m))[0]


@tf.function
def aug_zoom(image, m):
    shape = image.shape[:2]
    m = int((_MAX_LEVEL - m) / _MAX_LEVEL * image.shape[1])
    image = tf.image.random_crop(image, size=[m, m, image.shape[-1]])
    return tf.image.resize(image, shape)


@tf.function
def aug_saturation(image, m):
    return tf.image.random_saturation(image, 0.1, 0.3 * m + 0.1)


@tf.function
def aug_jpeg_quality(image, m):
    return tf.image.random_jpeg_quality(image,
                                        80 * (_MAX_LEVEL - m) // _MAX_LEVEL,
                                        90 * (_MAX_LEVEL - m) // _MAX_LEVEL)


def rand_augmentation(n, m, verbose=False):
    available_ops = [
        aug_contrast,
        aug_invert,
        aug_brightness,
        aug_rotate,
        # aug_shear_x,
        # aug_shear_y,
        aug_translate,
        aug_blur,
        aug_cutout,
        aug_zoom,
        aug_saturation,
        # aug_jpeg_quality
    ]

    def augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        for t in range(n):
            op_to_select = tf.random.uniform(
                [], maxval=len(available_ops), dtype=tf.int32)

            for (i, op_funct) in enumerate(available_ops):
                if verbose:
                    tf.cond(
                        tf.equal(i, op_to_select),
                        true_fn=lambda: tf.print(t, op_funct.__name__),
                        false_fn=lambda: True
                    )
                image = tf.cond(
                    tf.equal(i, op_to_select),
                    true_fn=lambda: op_funct(image, m),
                    false_fn=lambda: image
                )
        return image
    return augment
