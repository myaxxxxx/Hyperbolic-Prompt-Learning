export TASK_NAME=qa
export DATASET_NAME=squad
export CUDA_VISIBLE_DEVICES=2
export MODEL_PATH=pretraining_model/roberta-large
export EXP_NAME=squad_roberta_baselines
bs=8
lr=5e-3
dropout=0.2
psl=16
epoch=30

python3 run.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME/$EXP_NAME \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
