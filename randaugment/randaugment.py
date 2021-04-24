import tensorflow as tf
import tensorflow_addons as tfa

_MAX_LEVEL = 30

_PARAMS = {
    'blur_factor': 0.8,
    "brightness_base": 0.1,
    "brightness_factor": 1.4,
    "contrast_delta": 0.7,
    "cutout_factor": 0.4,
    "rotate_factor": 3.0,
    "saturation_lower": 7.0,
    "saturation_upper": 40.0,
    "translate_factor": 0.8
    # 'jpeg_quality_lower': 80,
    # 'jpeg_quality_lower': 100,
    # 'shear_factor': 0.3,
}


def rand_augmentation(n, m, verbose=False, params={}):
    PARAMS = _PARAMS
    PARAMS.update(params)
    print(PARAMS)

    @tf.function
    def aug_contrast(image):
        return tf.image.random_contrast(image,
                                        1 - PARAMS['contrast_delta'] * m / _MAX_LEVEL,
                                        1 + PARAMS['contrast_delta'] * m / _MAX_LEVEL)

    @tf.function
    def aug_invert(image):
        return 1 - image

    @tf.function
    def aug_transpose(image):
        shape = image.shape[:2]
        image = tf.image.transpose(image)
        return tf.image.resize(image, shape)

    @tf.function
    def aug_brightness(image):
        return tf.image.random_brightness(image, m / _MAX_LEVEL * PARAMS['brightness_factor'] + PARAMS['brightness_base'])

    @tf.function
    def aug_rotate(image):
        angles = PARAMS['rotate_factor'] * m / _MAX_LEVEL * \
            tf.random.uniform([], minval=-1, maxval=1, dtype=tf.float32)
        return tfa.image.rotate(image, angles=angles)

    @tf.function
    def aug_shear_x(image):
        color = [0, 0, 0]
        image = tfa.image.shear_x(
            image, m / _MAX_LEVEL * PARAMS['shear_factor'], replace=color)
        return image

    @tf.function
    def aug_shear_y(image):
        color = [0, 0, 0]
        image = tfa.image.shear_y(
            image, m / _MAX_LEVEL * PARAMS['shear_factor'], replace=color)
        return image

    @tf.function
    def aug_translate(image):
        return tfa.image.translate(image, PARAMS['translate_factor'] * image.shape[1] * m / _MAX_LEVEL * tf.random.uniform([2], minval=-1, maxval=1, dtype=tf.float32))

    @tf.function
    def aug_blur(image):
        return tfa.image.gaussian_filter2d(image, sigma=m / _MAX_LEVEL * PARAMS['blur_factor'])

    @tf.function
    def aug_cutout(image):
        dim = tf.cast(m / _MAX_LEVEL * PARAMS['cutout_factor'] * image.shape[1], tf.int32)
        dim = tf.bitwise.bitwise_and(dim, 1073741822)  # 1073741822 = 2^30 - 2
        image = tf.expand_dims(image, 0)
        return tfa.image.random_cutout(image, (dim, dim))[0]

    @tf.function
    def aug_zoom(image):
        shape = image.shape[:2]
        dim = tf.cast((_MAX_LEVEL - m) / _MAX_LEVEL * image.shape[1], tf.int64)
        image = tf.image.random_crop(image, size=[dim, dim, image.shape[-1]])
        return tf.image.resize(image, shape)

    @tf.function
    def aug_saturation(image):
        return tf.image.random_saturation(image,
                                          PARAMS['saturation_lower'] * (_MAX_LEVEL - m) / _MAX_LEVEL,
                                          1 + PARAMS['saturation_upper'] * (_MAX_LEVEL - m) / _MAX_LEVEL)

    @tf.function
    def aug_jpeg_quality(image):
        return tf.image.random_jpeg_quality(image,
                                            int(PARAMS['jpeg_quality_lower']) * (_MAX_LEVEL - m) // _MAX_LEVEL,
                                            int(PARAMS['jpeg_quality_upper']) * (_MAX_LEVEL - m) // _MAX_LEVEL)

    available_ops = [
        aug_contrast,
        aug_invert,
        aug_transpose,
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
                    true_fn=lambda: op_funct(image),
                    false_fn=lambda: image
                )
        return image
    return augment
