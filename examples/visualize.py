import randaugment as ra
import matplotlib.pyplot as plt
n = 5
m = 2
aug = ra.rand_augmentation(n=n, m=m, verbose=True)
dataset_train, dataset_test = ra.get_dataset_cifar100()
image, label = next(iter(dataset_train))
image = image.numpy()
print(image.shape)

for i in range(10):
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(image[0])
    axs[1].imshow(aug(image[0]))

    plt.show()
