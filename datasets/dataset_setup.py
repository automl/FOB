"""
Example command:

python3 datasets/dataset_setup.py \
    --data_dir=~/data \
    --mnist
"""
from absl import flags
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS

# general flags
flags.DEFINE_string(
    'data_dir',
    '~/data',
    'The path to the folder where datasets should be downloaded.')
flags.DEFINE_string(
    'temp_dir',
    '/tmp/bob',
    'A local path to a folder where temp files can be downloaded.')


# Workload flags
flags.DEFINE_boolean('mnist', False, 'Wheter to include mnist in download')
flags.DEFINE_boolean('all', False, 'Wheter to download all datasets.')

def download_mnist(data_dir):
    tfds.builder('mnist', data_dir=data_dir).download_and_prepare()


def main():
    if FLAGS.all or FLAGS.mnist:
        download_mnist()


if __name__ == '__main__':
    main()
