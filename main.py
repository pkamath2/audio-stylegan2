from argparse import ArgumentParser

from training.train import train
from util import util

def main(config):
    print('training')
    train(config, 0)


if __name__ == "__main__":
    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("--data_dir", dest="data_dir", help="Training data directory")
    parser.add_argument("--config_file", dest="config_file", default="config/config.json", help="Config JSON location with setting for stft_channels etc.,")
    parser.add_argument("--out_dir", dest="out_dir", help="Checkpoint directory")

    options = vars(parser.parse_args())
    config = util.get_config(options['config_file'])
    config = dict(**options, **config)

    print('Config ', config)
    main(config)