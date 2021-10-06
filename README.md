<h1 align="center">RandAugmentation</h1>

## About The Project
This project is a Tensorflow implementation of [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719).

## Getting Started
### Installation
1) Clone the repo 
```bash
git clone https://github.com/tpoppo/rand-augmentation.git
```
2) Install the package
```bash
pip install ./rand-augmentation-main
```

## Usage

```python3
import randaugment as ra
import tensorflow as tf

aug = ra.rand_augmentation(2, 5, verbose=False)
dataset_train, dataset_test = ra.get_dataset_cifar100(
    batch_size=1024,
    before_batch_train=lambda x, y: (aug(x), y))
model = ra.get_model_cifar100()

model.fit(dataset_train,
          validation_data=dataset_test,
          epochs=60
)
print(model.evaluate(dataset_test))
```
