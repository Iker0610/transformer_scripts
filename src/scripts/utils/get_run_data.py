import argparse
import json
import os
from glob import glob as get_files
from typing import Union

from pandas import DataFrame

# ROOT = '/gscratch4/users/idelaiglesia004/eriberta_evaluation/model_output/pharmaconer-bsc'

column_rename = {
    'eval_loss': 'Loss',
    'eval_micro_precision': 'Micro Precision',
    'eval_micro_recall': 'Micro Recall',
    'eval_micro_f1': 'Micro F1',
    'eval_macro_precision': 'Macro Precision',
    'eval_macro_recall': 'Macro Recall',
    'eval_macro_f1': 'Macro F1',
    'eval_overall_accuracy': 'Accuracy'
}

ordered_columns = [
    'dataset_split',
    'model',
    'learning_rate',
    'Loss',
    'Precision',
    'Recall',
    'F1',
    'Accuracy',
]


def get_experiment_results(input_folder: str, model, experiment) -> list[dict[str, Union[str, int]]]:
    results = list()

    for file in get_files(f'{input_folder}/{model}/{experiment}/*/eval_predictions_results.json'):
        with open(file, encoding='utf8') as file_hadler:
            results_json: dict[str, Union[str, int]] = json.load(file_hadler)
        results_json['model'] = model
        results_json['learning_rate'] = experiment
        results_json['dataset_split'] = 'dev'
        results.append(results_json)

    for file in get_files(f'{input_folder}/{model}/{experiment}/*/test_predictions_results.json'):
        with open(file, encoding='utf8') as file_hadler:
            results_json: dict[str, Union[str, int]] = json.load(file_hadler)
        results_json['model'] = model
        results_json['learning_rate'] = experiment
        results_json['dataset_split'] = 'test'
        results.append(results_json)

    return results


def get_model_results(input_folder: str, model: str) -> list[dict[str, Union[str, int]]]:
    experiments: list[str] = list(os.walk(f'{input_folder}/{model}'))[0][1]
    return [result for experiment in experiments for result in get_experiment_results(input_folder, model, experiment)]


def main(input_folder: str, output_file: str = './experiment_results.csv'):
    models = list(os.walk(input_folder))[0][1]
    results = [result for model in models for result in get_model_results(input_folder, model)]
    DataFrame(results).rename(columns=column_rename).to_csv(output_file, index_label='Index')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Experiment group folder.')
    parser.add_argument('-o', '--output_file', type=str, default='./experiment_results.csv', help='CSV file output file.')
    args = vars(parser.parse_args())

    main(**args)
