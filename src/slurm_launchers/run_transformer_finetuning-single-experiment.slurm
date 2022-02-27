#!/bin/bash

#SBATCH --job-name=MY_TRANSFORMER_eval
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=250M

#---------------------------------------------------------
# Preparar el entorno

source /var/python3envs/transformers-4.12.3/bin/activate

export HF_HOME=/gscratch/users/USER/MY_TRANSFORMER_evaluation
export TRANSFORMERS_CACHE=/gscratch/users/USER/MY_TRANSFORMER_evaluation/transformers

#---------------------------------------------------------
# Definir paths (a ser posible que sean absolutos)

OUTPUT_PATH='/gscratch/users/USER/MY_TRANSFORMER_evaluation/model_output/DATASET/MY_TRANSFORMER'

SRC_PATH='/tartalo03/users/udixa/ikasiker/MY_TRANSFORMER/MY_TRANSFORMER_evaluation/src/'
json_path="$SRC_PATH/model_params/DATASET/MY_TRANSFORMER_params.json"
train_script_path="$SRC_PATH/scripts/train_transformer_pytorch_allMetrics.py"
cleaner_script_path="$SRC_PATH/scripts/utils/experiment_cleaner.py"

#---------------------------------------------------------

# Definimos el learning rate
learning_rate="5e-5"

# Lanzamos el script de entrenamiento
python3.9 $train_script_path $json_path

# Limpiamos la carpeta output de los experimentos
python3.9 "$cleaner_script_path" -i "$OUTPUT_PATH/$learning_rate/"

# Copiamos el archivo con los parámetros del experimento a la carpeta del experimento para poder reproducirlo
cp $json_path "$OUTPUT_PATH/$learning_rate/params.json"