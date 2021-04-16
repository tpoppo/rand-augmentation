import randaugment as ra

# baseline
dataset_train, dataset_test = ra.get_dataset_cifar100(batch_size=512)
model = ra.get_model_cifar100()

model.fit(dataset_train, validation_data=dataset_test, epochs=60)
print(model.evaluate(dataset_test))
# with augmentation
aug = ra.rand_augmentation(2, 2, verbose=False)
dataset_train, dataset_test = ra.get_dataset_cifar100(
    batch_size=512,
    before_batch_train=lambda x, y: (aug(x), y))
model = ra.get_model_cifar100()

model.fit(dataset_train, validation_data=dataset_test, epochs=60)
print(model.evaluate(dataset_test))
