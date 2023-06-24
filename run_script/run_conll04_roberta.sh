# export TASK_NAME=ner
# export DATASET_NAME=conll2004
# export CUDA_VISIBLE_DEVICES=0
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=hy_roberta_baselines
# bs=32
# lr=6e-2
# dropout=0.1
# psl=144
# epoch=80

# python run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME/$EXP_NAME \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix




export TASK_NAME=ner
export DATASET_NAME=conll2004
export CUDA_VISIBLE_DEVICES=2
export MODEL_PATH=pretraining_model/roberta-large
export EXP_NAME=conll04_roberta_c
bs=32
lr=6e-2
dropout=0.1
psl=14
epoch=80


for i in 1 2 3 4 5 6 7 8 9 10;

do 

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
  --output_dir checkpoints/$DATASET_NAME/$EXP_NAME$i  \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --use_hy True --num_c $i


done 