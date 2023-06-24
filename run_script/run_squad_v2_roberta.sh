# export TASK_NAME=qa
# export DATASET_NAME=squad_v2
# export CUDA_VISIBLE_DEVICES=3
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=hy_test
# bs=8
# lr=5e-3
# dropout=0.2
# psl=8
# epoch=10

# python3 run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
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


# export TASK_NAME=qa
# export DATASET_NAME=squad_v2
# export CUDA_VISIBLE_DEVICES=1
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=hy_encoder_output_v2
# bs=8
# lr=5e-3
# dropout=0.2
# psl=8
# epoch=10

# nohup python run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
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
#   --prefix > $EXP_NAME.log



# export TASK_NAME=qa
# export DATASET_NAME=squad_v2
# export CUDA_VISIBLE_DEVICES=2
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=qa_baselines_v2
# bs=8
# lr=5e-3
# dropout=0.2
# psl=8
# epoch=10

# nohup python run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
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
#   --prefix > $EXP_NAME.log



# export TASK_NAME=qa
# export DATASET_NAME=squad_v2
# export CUDA_VISIBLE_DEVICES=1
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=qa_logit
# bs=8
# lr=5e-3
# dropout=0.2
# psl=8
# epoch=10

# nohup python run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
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
#   --prefix  > $EXP_NAME.log



# export TASK_NAME=qa
# export DATASET_NAME=squad_v2
# export CUDA_VISIBLE_DEVICES=2
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=qa_logit_c2
# bs=8
# lr=5e-3
# dropout=0.2
# psl=8
# epoch=10

# nohup python run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
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
#   --prefix  > $EXP_NAME.log



# export TASK_NAME=qa
# export DATASET_NAME=squad_v2
# export CUDA_VISIBLE_DEVICES=1
# export MODEL_PATH=pretraining_model/roberta-large
# export EXP_NAME=hy_encoder_output_c3
# bs=8
# lr=5e-3
# dropout=0.2
# psl=8
# epoch=10

# nohup python run.py \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
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
#   --prefix > $EXP_NAME.log





export TASK_NAME=qa
export DATASET_NAME=squad_v2
export CUDA_VISIBLE_DEVICES=2
export MODEL_PATH=pretraining_model/roberta-large
export EXP_NAME=hy_encoder_output_c4
bs=8
lr=5e-3
dropout=0.2
psl=8
epoch=10

nohup python run.py \
  --model_name_or_path $MODEL_PATH \
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
  --prefix > $EXP_NAME.log