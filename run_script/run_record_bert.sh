## 2023-5-8
# export TASK_NAME=superglue
# export DATASET_NAME=boolq
# export CUDA_VISIBLE_DEVICES=2
# export MODEL_PATH=pretraining_model/bert-large-uncased
# export EXP_NAME=hy_test

# bs=32
# lr=5e-3
# dropout=0.1
# psl=40
# epoch=100

# python run.py \
#   --model_name_or_path $MODEL_PATH  \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME-bert-$EXP_NAME/ \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix


# # 2023-5-9
# export TASK_NAME=superglue
# export DATASET_NAME=boolq
# export CUDA_VISIBLE_DEVICES=2
# export MODEL_PATH=pretraining_model/bert-large-uncased
# export EXP_NAME=hy_prompt_test

# bs=32
# lr=5e-3
# dropout=0.1
# psl=40
# epoch=100

# python run.py \
#   --model_name_or_path $MODEL_PATH  \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME-bert-$EXP_NAME/ \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prompt




## 2023-5-8
# export TASK_NAME=superglue
# export DATASET_NAME=boolq
# export CUDA_VISIBLE_DEVICES=1
# export MODEL_PATH=pretraining_model/bert-large-uncased
# export EXP_NAME=hy_bert_output

# bs=32
# lr=5e-3
# dropout=0.1
# psl=40
# epoch=100

# python run.py \
#   --model_name_or_path $MODEL_PATH  \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME-bert-$EXP_NAME/ \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix


# export TASK_NAME=superglue
# export DATASET_NAME=boolq
# export CUDA_VISIBLE_DEVICES=4
# export MODEL_PATH=pretraining_model/bert-large-uncased
# export EXP_NAME=hy_bert_c3

# bs=32
# lr=5e-3
# dropout=0.1
# psl=40
# epoch=100

# python run.py \
#   --model_name_or_path $MODEL_PATH  \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME-bert-$EXP_NAME/ \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix


# export TASK_NAME=superglue
# export DATASET_NAME=boolq
# export CUDA_VISIBLE_DEVICES=4
# export MODEL_PATH=pretraining_model/bert-large-uncased
# export EXP_NAME=hy_bert_c2

# bs=32
# lr=5e-3
# dropout=0.1
# psl=40
# epoch=100

# python run.py \
#   --model_name_or_path $MODEL_PATH  \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME-bert-$EXP_NAME/ \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix


export TASK_NAME=superglue
export DATASET_NAME=record
export CUDA_VISIBLE_DEVICES=4
export MODEL_PATH=pretraining_model/bert-large-uncased
export EXP_NAME=hy_bert_record

bs=32
lr=2e-5
dropout=0.3
psl=40
epoch=3

python run.py \
  --model_name_or_path $MODEL_PATH  \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-$EXP_NAME/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --num_c 6