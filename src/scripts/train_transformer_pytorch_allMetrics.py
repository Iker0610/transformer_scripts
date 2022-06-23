# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import transformers
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed, EarlyStoppingCallback,
)
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        metadata={"help": "Path to pretrained model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    vocab_path: Optional[str] = field(
        default=None, metadata={"help": "Vocab file path for the tokenizer."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.com"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    dev_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    dataset_loading_script: Optional[str] = field(
        default=None,
        metadata={"help": "Path to an optional dataset_processing_script."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path where dataset cache will be saved."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
                    "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    use_sliding_window: bool = field(
        default=True,
        metadata={"help": "If true it will use a sliding window in order to classify all the corpus. If False it will truncate overflowing tokens."}
    )

    stride_size: int = field(
        default=0,
        metadata={
            "help": "When sentences overflow and are added as new sentences, the amount of tokens that will be strided."
        }
    )

    def __post_init__(self):
        if self.train_file is None and self.dev_file is None and self.test_file is None:
            raise ValueError("Need a training/dev/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "conll"], "`train_file` should be a conll, csv or a json file."

            if self.dev_file is not None:
                extension = self.dev_file.split(".")[-1]
                assert extension in ["csv", "json", "conll"], "`dev_file` should be a conll, csv or a json file."

            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "conll"], "`test_file` should be a conll, csv or a json file."

        self.task_name = self.task_name.lower()


@dataclass
class EarlyStoppingArguments:
    """
    Arguments pertaining to early stopping configuration
    """
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Activate early stopping. It requires save_best_model to be set and set everything to steps instead of epochs."},
    )

    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls."},
    )

    early_stopping_threshold: float = field(
        default=0.0001,
        metadata={"help": "Use with TrainingArguments metric_for_best_model and early_stopping_patience to denote how much the specified metric must improve to satisfy early stopping conditions."},
    )


@dataclass
class HyperparameterOptimizationArguments:
    """
    Arguments pertaining to hyperparameter optimization configuration
    """
    do_hyperparameter_optimization: bool = field(
        default=False,
        metadata={"help": "Activate hyperparameter optimization using raytune and population-based optimization."}
    )

    ray_directory: str = field(
        default='~/ray_tune'
    )

    ray_experiment_name: str = field(
        default='tune_transformer_pbt'
    )

    ray_cpu_number: int = field(
        default=1
    )

    ray_gpu_number: int = field(
        default=0
    )

    keep_checkpoints_num: int = field(
        default=1
    )

    n_trials: int = field(
        default=10,
        metadata={"help": " The number of trial runs with different hyperparameters to test."}
    )

    smoke_test: bool = field(
        default=False
    )

    perturbation_interval: float = field(
        default=1
    )

    burn_in_period: float = field(
        default=0
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, EarlyStoppingArguments, HyperparameterOptimizationArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args: ModelArguments
        data_args: DataTrainingArguments
        training_args: TrainingArguments
        early_stopping_args: EarlyStoppingArguments
        hp_tune_args: HyperparameterOptimizationArguments

        model_args, data_args, training_args, early_stopping_args, hp_tune_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        model_args, data_args, training_args, early_stopping_args, hp_tune_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load training callbacks
    callbacks = []

    if early_stopping_args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_args.early_stopping_patience,
            early_stopping_threshold=early_stopping_args.early_stopping_threshold,
        ))

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = {}
    if training_args.do_train or hp_tune_args.do_hyperparameter_optimization:
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        else:
            raise ValueError("Need a training file for training or hyperparameter optimization.")

    if training_args.do_eval or hp_tune_args.do_hyperparameter_optimization:
        if data_args.dev_file is not None:
            data_files["dev"] = data_args.dev_file
        else:
            raise ValueError("Need a dev file for evaluating or hyperparameter optimization.")

    if training_args.do_predict:
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need a test file for predicting.")

    if data_args.dataset_loading_script is not None:
        loading_script = data_args.dataset_loading_script
    else:

        loading_script = data_args.train_file.split(".")[-1]

    # Nos aseguramos que el path al cache existe:
    if data_args.dataset_cache_dir is not None and not os.path.exists(data_args.dataset_cache_dir):
        os.makedirs(data_args.dataset_cache_dir)

    # Generamos los datasets
    datasets = load_dataset(loading_script, data_files=data_files, cache_dir=data_args.dataset_cache_dir)

    if training_args.do_train or hp_tune_args.do_hyperparameter_optimization:
        column_names = datasets["train"].column_names
        features = datasets["train"].features

    elif training_args.do_eval or hp_tune_args.do_hyperparameter_optimization:
        column_names = datasets["dev"].column_names
        features = datasets["dev"].features

    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[-1]
    )

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(p_labels):
        unique_labels = set()
        for label in p_labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}

    elif training_args.do_train or hp_tune_args.do_hyperparameter_optimization:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    else:
        raise ValueError("Need a label list to evaluate and to predict.")

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # Cargamos la configuración del modelo:
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    # Se carga el modelo
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(
            model_args.model_path,
            from_tf=bool(".ckpt" in model_args.model_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Cargamos el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        config=config,
        add_prefix_space=True
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        window_size = data_args.stride_size  # int(data_args.max_seq_length / 2)
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            # Estas opciones son vitales para hacer el sliding window
            return_offsets_mapping=True,
            return_overflowing_tokens=data_args.use_sliding_window,
            stride=window_size,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        if not data_args.use_sliding_window:
            tokenized_inputs['overflow_to_sample_mapping'] = list(range(0, len(tokenized_inputs['input_ids'])))

        # Aquí guardamos el índice de documento de cada fragmento.
        # Si un documento (ej el nº 12) se ha dividido en 3 entonces cada fragmento tendrá el mismo id (12)
        previous_doc_index = None

        # Iteramos sobre los fragmentos
        for file_fragment_index in range(0, len(tokenized_inputs['input_ids'])):

            # Obtenemos el indice del documento del feagmento
            original_file_index = tokenized_inputs['overflow_to_sample_mapping'][file_fragment_index]
            # Obtenemos el índice de palabras del fragmento; este índice es respecto al texto original y la palabra original (en caso de que al tokenizar la palabra se fragmente)
            word_ids = tokenized_inputs.word_ids(batch_index=file_fragment_index)

            # Generamos valores para indicar los índices de las palabras
            first_word_indx = None
            last_word_indx = None

            # Si hemos cambiado de documento entonces reseteamos los valores
            if previous_doc_index != original_file_index:
                previous_word_idx = None

            # Si no los ajustamos según el sliding window
            else:
                # Al tratarse de un fragmento se asume que el fragmento anterior es completo (no padding)
                previous_word_idx = tokenized_inputs.word_ids(batch_index=file_fragment_index - 1)[data_args.max_seq_length - (2 + window_size)]

            # Obtenemos las anotaciones
            file_labels = examples[label_column_name][original_file_index]
            # Generamos una lista para cada label de cada subtoken. Los tokens ignorados tendrán un -100
            label_ids = []

            # Iteramos sobre los word ids para asignarles un label
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)

                # Si no es un special token (start, end, padding, ...) le asignamos una etiqueta según la configuración
                else:

                    if word_idx != previous_word_idx or data_args.label_all_tokens:
                        if first_word_indx is None:
                            first_word_indx = word_idx
                        last_word_indx = word_idx

                    # We set the label for the first token of each word.
                    if word_idx != previous_word_idx:
                        label_ids.append(label_to_id[file_labels[word_idx]])

                    # For all the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(label_to_id[file_labels[word_idx]] if data_args.label_all_tokens else -100)

                    previous_word_idx = word_idx

            tokenized_inputs.setdefault('overflow_to_sample_index_mapping', []).append((first_word_indx, last_word_indx))
            tokenized_inputs.setdefault('labels', []).append(label_ids)

            for key in examples.keys():
                if any(isinstance(elem, list) for elem in examples[key]):
                    if last_word_indx is not None:
                        file_fragment_data = examples[key][original_file_index][first_word_indx:last_word_indx + 1]
                    else:
                        file_fragment_data = []
                else:
                    file_fragment_data = examples[key][original_file_index]
                tokenized_inputs.setdefault(key, []).append(file_fragment_data)

            # Actualizamos el índice del documento previo
            previous_doc_index = original_file_index

        return tokenized_inputs

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Metrics
    metric = load_metric("../seqeval_allMetrics.py")

    def compute_metrics(p, are_predictions_processed=False):
        if not are_predictions_processed:
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
        else:
            true_predictions, true_labels = p

        results = metric.compute(predictions=true_predictions, references=true_labels)

        # Unpack nested dictionaries
        final_results = {}

        for key, value in results.items():
            if isinstance(value, dict):
                if data_args.return_entity_level_metrics:
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = float(v)
            else:
                final_results[key] = float(value)
        return final_results

    if hp_tune_args.do_hyperparameter_optimization:
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model_init(),
            args=training_args,
            train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
            eval_dataset=tokenized_datasets["dev"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def predict_and_save_to_conll(prediction_dataset: str, output_file: str):

        if trainer.is_world_process_zero() and data_args.task_name == 'ner':

            prediction_dataset = tokenized_datasets[prediction_dataset]
            output_predictions_file = os.path.join(training_args.output_dir, output_file)

            predictions, labels, prediction_results = trainer.predict(prediction_dataset, metric_key_prefix='eval')

            # Guardar resultados en conll
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            # Save predictions
            merged_true_predictions: list[list] = []
            merged_true_labels: list[list] = []

            previous_file_indx = None
            previous_file_last_token = None

            with open(f'{output_predictions_file}.conll', "w", encoding='utf8') as writer:
                for file_id, file_indx, file_token_index_range, *token_data in zip(prediction_dataset['id'],
                                                                                   prediction_dataset['overflow_to_sample_mapping'],
                                                                                   prediction_dataset['overflow_to_sample_index_mapping'],
                                                                                   # The next is saved in *token_data
                                                                                   prediction_dataset['tokens'],
                                                                                   prediction_dataset['line_offset'],
                                                                                   true_predictions, true_labels):

                    if file_token_index_range[0] is None or file_token_index_range[1] is None:
                        continue

                    if file_indx != previous_file_indx:
                        if previous_file_indx is not None:
                            writer.write('\n\n')
                        writer.write(f'FILE {file_id} -\n')

                        previous_file_indx = file_indx
                        good_token_start = None

                        merged_true_predictions[file_indx] = []
                        merged_true_labels[file_indx] = []

                    else:
                        good_token_start = (previous_file_last_token + 1) - file_token_index_range[0]

                    for token, offset, prediction, label in list(zip(*token_data))[good_token_start:]:
                        writer.write(f'{token} {offset} {prediction}\n')

                        merged_true_predictions[file_indx].append(prediction)
                        merged_true_labels[file_indx].append(label)

                    previous_file_last_token = file_token_index_range[1]

            # Make sliding window do not have advantage: update prediction_results with new metrics obtained from compute_metrics
            prediction_results.update(
                compute_metrics((merged_true_predictions, merged_true_labels), are_predictions_processed=True)
            )

            # Log evaluation
            logger.info("***** Eval results *****")
            for key, value in prediction_results.items():
                logger.info(f"  {key} = {value}")

            # Save evaluation in json
            with open(f'{output_predictions_file}_results.json', "w", encoding='utf8') as writer:
                json.dump(prediction_results, writer)

    # Hyperparameter Optimization
    if hp_tune_args.do_hyperparameter_optimization:
        # Imports
        from ray import tune
        from ray.tune import CLIReporter
        from ray.tune.schedulers import PopulationBasedTraining

        # Deshabilitamos las barras de cargas (dan problemas en windows)
        training_args.disable_tqdm = True

        # Configuramos algunos parámetros iniciales
        tune_config = {
            "max_steps": 1 if hp_tune_args.smoke_test else -1,  # Used for smoke test.
            "seed": tune.uniform(1, 40),
        }

        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric=f'eval_{training_args.metric_for_best_model}',
            mode="max" if training_args.greater_is_better else "min",
            perturbation_interval=hp_tune_args.perturbation_interval,
            burn_in_period=hp_tune_args.burn_in_period,
            hyperparam_mutations={
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.uniform(1e-5, 5e-5),
            },
        )

        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
            },
            metric_columns=["eval_micro_f1", "eval_loss", "epoch", "training_iteration"],
        )

        trainer.hyperparameter_search(
            backend="ray",
            hp_space=lambda _: tune_config,
            scheduler=scheduler,
            n_trials=hp_tune_args.n_trials,
            resources_per_trial={"cpu": hp_tune_args.ray_cpu_number, "gpu": hp_tune_args.ray_gpu_number},
            keep_checkpoints_num=hp_tune_args.keep_checkpoints_num,
            stop={"training_iteration": 1} if hp_tune_args.smoke_test else None,
            progress_reporter=reporter,
            local_dir=hp_tune_args.ray_directory,
            name=hp_tune_args.ray_experiment_name,
            log_to_file=True,
        )

    # Training
    if training_args.do_train and not hp_tune_args.do_hyperparameter_optimization:
        train_result = trainer.train(
            model_path=model_args.model_path if os.path.isdir(model_args.model_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w", encoding='utf8') as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        predict_and_save_to_conll(prediction_dataset='dev', output_file='eval_predictions')

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_and_save_to_conll(prediction_dataset='test', output_file='test_predictions')

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
