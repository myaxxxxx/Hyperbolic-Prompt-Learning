export TASK_NAME=ner
export DATASET_NAME=ontonotes
export CUDA_VISIBLE_DEVICES=0


export MODEL_PATH=pretraining_model/roberta-large
export EXP_NAME=hy_roberta_onenote_baseline

bs=16
lr=7e-3
dropout=0.1
psl=48
epoch=60

python3 run.py \
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
  --output_dir checkpoints/$DATASET_NAME-roberta/$EXP_NAME \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 14 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --use_hy False
