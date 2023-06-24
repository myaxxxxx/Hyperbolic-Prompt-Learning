export TASK_NAME=srl
export DATASET_NAME=conll2012
export CUDA_VISIBLE_DEVICES=0

export MODEL_PATH=pretraining_model/roberta-large
export EXP_NAME=hy_roberta_baselines
bs=32
lr=5e-3
dropout=0.1
psl=64
epoch=45


python run.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
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
