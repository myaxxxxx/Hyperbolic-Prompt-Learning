export TASK_NAME=ner
export DATASET_NAME=conll2003
export CUDA_VISIBLE_DEVICES=0

export EXP_NAME=roberta_conll03_pt2_baselines
bs=16
epoch=30
psl=11
lr=3e-2
dropout=0.1



# nohup 
python3 run.py \
  --model_name_or_path /data02/GaoGL_GRP/GglStuA/student/pt2/pretraining_model/roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 152 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME/roberta_conll03_pt2_baselines \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --use_hy False
  
  #  > $EXP_NAME$i.log



# for i in 1 2 3 4 5 6 7 8 9 10;

# do 

# nohup python3 run.py \
#   --model_name_or_path /data02/GaoGL_GRP/GglStuA/student/pt2/pretraining_model/roberta-large \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --max_seq_length 152 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME/roberta_conll03_ft_c$i \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --ft --use_hy True --num_c $i  > $EXP_NAME$i.log
# done 
# for i in 1 2 3 4 5 6 7 8 9 10;

# do 

# nohup python3 run.py \
#   --model_name_or_path /data02/GaoGL_GRP/GglStuA/student/pt2/pretraining_model/roberta-large \
#   --task_name $TASK_NAME \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --max_seq_length 152 \
#   --per_device_train_batch_size $bs \
#   --learning_rate $lr \
#   --num_train_epochs $epoch \
#   --pre_seq_len $psl \
#   --output_dir checkpoints/$DATASET_NAME/$EXP_NAME$i \
#   --overwrite_output_dir \
#   --hidden_dropout_prob $dropout \
#   --seed 11 \
#   --save_strategy no \
#   --evaluation_strategy epoch \
#   --prefix --use_hy True --num_c $i > $EXP_NAME$i.log


# done 
