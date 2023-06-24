export TASK_NAME=superglue
export DATASET_NAME=copa
export CUDA_VISIBLE_DEVICES=5
export EXP_NAME=copa_c
export MODEL_PATH=pretraining_model/bert-large-uncased

bs=16
lr=1e-2
dropout=0.1
psl=16
epoch=80



# for i in 1 2 3 4 5 6 7 8 9 10;
for i in 0.1 0.01 0.001 3.5 4.5 2.5 5.5 ;
do
echo $i 
nohup python3 run.py \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert/$EXP_NAME$i \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --use_hy True --num_c $i > $EXP_NAME$i.log


done
