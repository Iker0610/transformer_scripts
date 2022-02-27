#!/bin/bash

#SBATCH --job-name=eriberta_pharmaconer_eval
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=250M

#---------------------------------------------------------
# Preparar el entorno

source /var/python3envs/transformers-4.12.3/bin/activate

export HF_HOME=/gscratch/users/idelaiglesia004/eriberta_evaluation
export TRANSFORMERS_CACHE=/gscratch/users/idelaiglesia004/eriberta_evaluation/transformers

#---------------------------------------------------------
# Definir paths (a ser posible que sean absolutos)

OUTPUT_PATH='/gscratch/users/idelaiglesia004/eriberta_evaluation/model_output/pharmaconer-bsc/eriberta'

SRC_PATH='/tartalo03/users/udixa/ikasiker/Eriberta/eriberta_evaluation/src/'
json_path="$SRC_PATH/model_params/pharmaconer-bsc/bio-cli_params.json"
train_script_path="$SRC_PATH/scripts/train_transformer_Pytorch.py"
cleaner_script_path="$SRC_PATH/scripts/experiment_cleaner.py"

#---------------------------------------------------------

# Definimos la función encargada de hacer los experimentos
function train-transformer() {
  # Obtenemos el learning rate que se obtendrá por parámetro
  learning_rate=$1

  # Hacemos X pruebas
  for i in {1..10}; do

    # Copiamos la plantilla de los parámetros que hay que pasarle al script de entrenamiento
    cp "$SRC_PATH/model_params/pharmaconer-bsc/plantilla_eriberta_params.json" "$json_path"

    # Generamos una seed random
    seed=$(shuf -i 0-5500 -n 1)

    # Editamos la COPIA de la plantilla poniendo los parámetros correctos
    sed -i.bak "s#prueba_X#prueba_$i#" $json_path
    sed -i.bak "s#VAR_LR#$learning_rate#" $json_path
    sed -i.bak "s#\"seed\": VAR_SEED#\"seed\": $seed#" $json_path

    # Lanzamos el script de entrenamiento
    python3.9 $train_script_path $json_path

    # Limpiamos las carpetas output de los experimentos
    python3.9 "$cleaner_script_path" -i "$OUTPUT_PATH/$learning_rate/prueba_$i" --delete_best_model

    # Copiamos el archivo con los parámetros del experimento a la carpeta del experimento para poder reproducirlo
    cp $json_path "$OUTPUT_PATH/$learning_rate/prueba_$i/params.json"
  done
}

#---------------------------------------------------------
# Definir que experimentos hacer con qué learning rates (referencia: el por defecto de HuggingFace es 5e-5)

train-transformer "5e-5"
#train-transformer "75e-6"
#train-transformer "1e-4"
#train-transformer "125e-6"
