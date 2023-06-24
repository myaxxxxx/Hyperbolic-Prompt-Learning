export TASK_NAME=superglue
export DATASET_NAME=wsc
export CUDA_VISIBLE_DEVICES=3
export MODEL_PATH=pretraining_model/bert-large-uncased
export EXP_NAME=hy_bert_wsc_c


bs=16
lr=5e-3
dropout=0.1
psl=20
epoch=80

# i=1
for i in 0.001 0.0001 0.00001 ;  
do  

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
  --output_dir checkpoints/$DATASET_NAME/$EXP_NAME$i \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --num_c $i --use_hy True > $EXP_NAME$i.log

done