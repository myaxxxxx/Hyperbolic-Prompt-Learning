# export TASK_NAME=ner
# export DATASET_NAME=conll2004
# export CUDA_VISIBLE_DEVICES=1
# export EXP_NAME=baselines_test
# export MODEL_PATH=pretraining_model/bert-large-uncased
# bs=32
# lr=2e-2
# dropout=0.2
# psl=128
# epoch=40

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


## best
# export TASK_NAME=ner
# export DATASET_NAME=conll2004
# export CUDA_VISIBLE_DEVICES=3
# export EXP_NAME=hy_bert_output_test2
# export MODEL_PATH=pretraining_model/bert-large-uncased
# bs=32
# lr=2e-2
# dropout=0.2
# psl=128
# epoch=40

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
export CUDA_VISIBLE_DEVICES=0
export EXP_NAME=hy_bert_baseline
export MODEL_PATH=pretraining_model/bert-large-uncased
bs=32
lr=2e-2
dropout=0.2
psl=128
epoch=40

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