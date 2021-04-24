import randaugment as ra
import tensorflow as tf

'''
# baseline
dataset_train, dataset_test = ra.get_dataset_cifar100(batch_size=1024)
model = ra.get_model_cifar100()

model.fit(dataset_train,
          validation_data=dataset_test,
          epochs=60,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                      mode='max')])
print(model.evaluate(dataset_test))
'''
# with augmentation
aug = ra.rand_augmentation(2, 5, verbose=False)
dataset_train, dataset_test = ra.get_dataset_cifar100(
    batch_size=1024,
    before_batch_train=lambda x, y: (aug(x), y))
model = ra.get_model_cifar100()

model.fit(dataset_train,
          validation_data=dataset_test,
          epochs=60,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                      mode='max')])
print(model.evaluate(dataset_test))
