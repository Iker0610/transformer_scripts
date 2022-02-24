import json
import os
from typing import Union
from glob import glob as get_files

from pandas import DataFrame

ROOT = '/tartalo03/users/udixa/ikasiker/Eriberta/eriberta_evaluation/results/pharmaconer-bsc'

column_rename = {
    'eval_loss': 'Loss',
    'eval_precision': 'Precision',
    'eval_recall': 'Recall',
    'eval_f1': 'F1',
    'eval_accuracy': 'Accuracy'
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


def get_experiment_results(model, experiment) -> list[dict[str, Union[str, int]]]:
    results = list()

    for file in get_files(f'{ROOT}/{model}/{experiment}/*/eval_predictions_results.json'):
        with open(file, encoding='utf8') as file_hadler:
            results_json: dict[str, Union[str, int]] = json.load(file_hadler)
        results_json['model'] = model
        results_json['learning_rate'] = experiment
        results_json['dataset_split'] = 'dev'
        results.append(results_json)

    for file in get_files(f'{ROOT}/{model}/{experiment}/*/test_predictions_results.json'):
        with open(file, encoding='utf8') as file_hadler:
            results_json: dict[str, Union[str, int]] = json.load(file_hadler)
        results_json['model'] = model
        results_json['learning_rate'] = experiment
        results_json['dataset_split'] = 'test'
        results.append(results_json)

    return results


def get_model_results(model: str) -> list[dict[str, Union[str, int]]]:
    experiments: list[str] = list(os.walk(f'{ROOT}/{model}'))[0][1]
    return [result for experiment in experiments for result in get_experiment_results(model, experiment)]


def main():
    models = list(os.walk(ROOT))[0][1]
    results = [result for model in models for result in get_model_results(model)]
    DataFrame(results).rename(columns=column_rename).to_csv('/tartalo03/users/udixa/ikasiker/Eriberta/eriberta_evaluation/results/pharmaconer-bsc/experiment_results.csv', index_label='Index', columns=ordered_columns)


if __name__ == '__main__':
    main()
