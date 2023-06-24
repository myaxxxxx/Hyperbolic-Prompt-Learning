export TASK_NAME=srl
export DATASET_NAME=conll2012

export CUDA_VISIBLE_DEVICES=3
export EXP_NAME=hy_bert_conll12_c
export MODEL_PATH=pretraining_model/bert-large-uncased
bs=8
lr=5e-3
dropout=0.1
psl=128
epoch=45
# for i in 1 2 3 4 5;

# do 
# nohup 
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
  --output_dir checkpoints/$DATASET_NAME/$EXP_NAME$i \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix --use_hy True 
  # > $EXP_NAME.log

# done 