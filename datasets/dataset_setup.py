"""
Example command:

python3 datasets/dataset_setup.py \
    --data_dir=~/data \
    --mnist
"""
import sys
from absl import flags
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


# general flags
flags.DEFINE_string(
    'data_dir',
    '~/data_bob',
    'The path to the folder where datasets should be downloaded.')
flags.DEFINE_string(
    'temp_dir',
    '/tmp/bob',
    'A local path to a folder where temp files can be downloaded.')

# Workload flags
flags.DEFINE_boolean('mnist', False, 'Whether to include MNIST in download')
flags.DEFINE_boolean('cifar10', False, 'Whether to include CIFAR-10 in download')
flags.DEFINE_boolean('cifar100', False, 'Whether to include CIFAR-100 in download')
flags.DEFINE_boolean('all', False, 'Whether to download all datasets.')

FLAGS = flags.FLAGS

def download_mnist(data_dir):
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    # training set
    MNIST(root=data_dir, train=True, download=True)
    # test set
    MNIST(root=data_dir, train=False, download=True)


def download_cifar10(data_dir):
    # https://pytorch.org/vision/0.8/datasets.html#cifar
    # training set
    CIFAR10(root=data_dir, train=True, download=True)
    # test set
    CIFAR10(root=data_dir, train=False, download=True)


def download_cifar100(data_dir):
    # https://pytorch.org/vision/0.8/datasets.html#cifar
    # training set
    CIFAR100(root=data_dir, train=True, download=True)
    # test set
    CIFAR100(root=data_dir, train=False, download=True)


def main():
    # need to explicitly tell flags library to parse argv before you can access FLAGS.xxx
    # we could do this implicitly by using app.run()
    FLAGS(sys.argv)

    if FLAGS.all or FLAGS.mnist:
        download_mnist(FLAGS.data_dir)
    if FLAGS.all or FLAGS.cifar10:
        download_cifar10(FLAGS.data_dir)
    if FLAGS.all or FLAGS.cifar100:
        download_cifar100(FLAGS.data_dir)

if __name__ == '__main__':
    main()
