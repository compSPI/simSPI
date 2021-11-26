import argparse
import os
import sys
import traceback

from .config import Config as cfg
from .dataset_generator import dataset_gen
from .params_utils import params_update


# TODO: add pbar
def init_config():
    # takes the cfg file with parameters and creates a variable called config with those parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Specify config file", metavar="FILE")
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError("Please provide a valid .cfg file")

    config = cfg(args.config)
    config = params_update(config)
    return config


def main():
    config = init_config()
    dataset_generator = dataset_gen(config)
    dataset_generator.run()
    return 0, "Dataset successfully generated."


if __name__ == "__main__":
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = "Error: Training failed."

    print(status_message)
    exit(retval)
