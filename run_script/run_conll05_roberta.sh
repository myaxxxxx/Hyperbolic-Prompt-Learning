export TASK_NAME=srl
export DATASET_NAME=conll2005
export CUDA_VISIBLE_DEVICES=4
export MODEL_PATH=pretraining_model/roberta-large
export EXP_NAME=hy_roberta_c
bs=16
lr=6e-3
dropout=0.1
psl=224
epoch=15


for i in 0.1 0.01 0.001 0.5;
do
nohup python3 run.py \
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
  --output_dir checkpoints/$DATASET_NAME/$EXP_NAME$i \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --use_hy --num_c $i > $EXP_NAME$i.log

done