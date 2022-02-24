import os
import shutil
import sys
from glob import glob as get_files

FILES_TO_REMOVE = [
    'config.json',
    'pytorch_model.bin',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'train_results.txt',
    'vocab.txt',
    'vocab.json',
    'merges.txt',
    'training_args.bin',
]


def clean_path(folder_path: str):
    for file in FILES_TO_REMOVE:
        try:
            os.remove(f'{folder_path}/{file}')
        except FileNotFoundError as e:
            pass

    for experiment_folder in get_files(f'{folder_path}/checkpoint*'):
        shutil.rmtree(experiment_folder)


if __name__ == '__main__':
    for folder in get_files(sys.argv[1]):
        clean_path(folder)
