import randaugment as ra
import tensorflow as tf
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer


def training_function(**config):

    aug = ra.rand_augmentation(2, 5, verbose=False, params=config)
    dataset_train, dataset_test = ra.get_dataset_cifar100(
        batch_size=1024,
        before_batch_train=lambda x, y: (aug(x), y))
    model = ra.get_model_cifar100()
    history = model.fit(dataset_train,
                        validation_data=dataset_test,
                        verbose=0,
                        epochs=100,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max')]).history

    return max(history['val_sparse_categorical_accuracy'])


bounds_transformer = SequentialDomainReductionTransformer()

optimizer = BayesianOptimization(
    f=training_function,
    verbose=2,
    random_state=42,
    # bounds_transformer=bounds_transformer,
    pbounds={
        'contrast_delta': (0.1, 0.8),
        'brightness_factor': (0.5, 2.0),
        'brightness_base': (0.01, 0.4),
        'rotate_factor': (0.1, 3.5),
        'translate_factor': (0.2, 0.8),
        # 'blur_factor': (1.2, 1.2),
        'cutout_factor': (0.3, 1.0),
        'saturation_lower': (0, 8),
        'saturation_upper': (9, 90),
        # 'jpeg_quality_lower': (40, 80),
        # 'jpeg_quality_lower': (80, 100),
        # 'shear_factor': (0.1, 0.5),
    }
)
try:
    optimizer.maximize(
        init_points=5,
        n_iter=120,
    )
except KeyboardInterrupt:
    pass

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
print("\nBest {}".format(optimizer.max))
