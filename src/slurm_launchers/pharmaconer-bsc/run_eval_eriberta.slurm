#!/bin/bash

#SBATCH --job-name=eriberta_pharmaconer_eval
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=250M

#---------------------------------------------------------

source /var/python3envs/transformers-4.12.3/bin/activate

export HF_HOME=/gscratch/users/idelaiglesia004/eriberta_evaluation
export TRANSFORMERS_CACHE=/gscratch/users/idelaiglesia004/eriberta_evaluation/transformers

#---------------------------------------------------------

OUTPUT_PATH='/gscratch/users/idelaiglesia004/eriberta_evaluation/model_output/pharmaconer-bsc/eriberta'

SRC_PATH='/tartalo03/users/udixa/ikasiker/Eriberta/eriberta_evaluation/src/'
json_path="$SRC_PATH/model_params/pharmaconer-bsc/bio-cli_params.json"
train_script_path="$SRC_PATH/scripts/train_transformer_Pytorch.py"
cleaner_script_path="$SRC_PATH/scripts/experiment_cleaner.py"

#---------------------------------------------------------

cp "$SRC_PATH/model_params/pharmaconer-bsc/plantilla_eriberta_params.json" "$json_path"

#---------------------------------------------------------

function train-transformer() {
  learning_rate=$1

  for i in {1..10}; do
    seed=$(shuf -i 0-5500 -n 1)

    sed -i.bak "s#prueba_X#prueba_$i#" $json_path
    sed -i.bak "s#VAR_LR#$learning_rate#" $json_path
    sed -i.bak "s#\"seed\": VAR_SEED#\"seed\": $seed#" $json_path

    python3.9 $train_script_path $json_path
    python3.9 "$cleaner_script_path" "$OUTPUT_PATH/$learning_rate/prueba_$i"

    cp $json_path "$OUTPUT_PATH/$learning_rate/prueba_$i/params.json"

    sed -i.bak "s#prueba_$i#prueba_X#" $json_path
    sed -i.bak "s#$learning_rate#VAR_LR#" $json_path
    sed -i.bak "s#\"seed\": $seed#\"seed\": VAR_SEED#" $json_path
  done
}

train-transformer "5e-5"
#train-transformer "75e-6"
#train-transformer "1e-4"
#train-transformer "125e-6"
