import argparse
import json
from pprint import pprint
from typing import Optional
from glob import glob as get_files


def get_conll_dataset_tags(input_folder: str, num_columns: int, outpur_file: Optional[str], separator: str = ' ', tag_column_index: int = -1):
    file_lines: list[str] = []
    for file in get_files(f'{input_folder}/**/*.conll', recursive=True):
        with open(file, encoding='utf8') as file_handler:
            file_lines += file_handler.read().splitlines()

    unique_labels: list[str] = list(set([line.split(separator)[tag_column_index] for line in file_lines if line and len(line.split(separator)) == num_columns]))
    pprint(unique_labels)

    if outpur_file:
        with open(outpur_file, 'w', encoding='utf-8') as output_file:
            json.dump(unique_labels, output_file, ensure_ascii=False, indent=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Folder with conll files (conll may be in subfolders).')
    parser.add_argument('-o', '--outpur_file', type=Optional[str], default=None, help='JSON file to save the list of labels.')
    parser.add_argument('-s', '--separator', type=str, default=' ', help='CONLL files\' column separator.')
    parser.add_argument('-c', '--num_columns', type=int, required=True, help='Number of columns conll files have. (needed for checking)')
    parser.add_argument('-t', '--tag_column_index', type=int, default=-1, help='Column index where CONLL tags are located.')
    args = parser.parse_args()

    get_conll_dataset_tags(**vars(args))
