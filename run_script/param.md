usage: run.py [-h] --model_name_or_path MODEL_NAME_OR_PATH [--config_name CONFIG_NAME] [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR] [--no_use_fast_tokenizer]
              [--use_fast_tokenizer [USE_FAST_TOKENIZER]] [--model_revision MODEL_REVISION] [--use_auth_token [USE_AUTH_TOKEN]] [--prefix [PREFIX]] [--prompt [PROMPT]] [--pre_seq_len PRE_SEQ_LEN]
              [--prefix_projection [PREFIX_PROJECTION]] [--prefix_hidden_size PREFIX_HIDDEN_SIZE] [--no_use_hyperbolic] [--use_hyperbolic [USE_HYPERBOLIC]] [--hidden_dropout_prob HIDDEN_DROPOUT_PROB]
              --task_name {glue,superglue,ner,srl,qa} --dataset_name
              {cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb,wnli,boolq,cb,rte,wic,wsc,copa,record,multirc,conll2003,conll2004,ontonotes,conll2005,conll2012,squad,squad_v2}
              [--dataset_config_name DATASET_CONFIG_NAME] [--max_seq_length MAX_SEQ_LENGTH] [--overwrite_cache [OVERWRITE_CACHE]] [--no_pad_to_max_length] [--pad_to_max_length [PAD_TO_MAX_LENGTH]]
              [--max_train_samples MAX_TRAIN_SAMPLES] [--max_eval_samples MAX_EVAL_SAMPLES] [--max_predict_samples MAX_PREDICT_SAMPLES] [--train_file TRAIN_FILE] [--validation_file VALIDATION_FILE]
              [--test_file TEST_FILE] [--template_id TEMPLATE_ID] --output_dir OUTPUT_DIR [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]] [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
              [--do_predict [DO_PREDICT]] [--evaluation_strategy {no,steps,epoch}] [--prediction_loss_only [PREDICTION_LOSS_ONLY]] [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
              [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE] [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE] [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
              [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
              [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM] [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS]
              [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}] [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS]
              [--log_level {debug,info,warning,error,critical,passive}] [--log_level_replica {debug,info,warning,error,critical,passive}] [--no_log_on_each_node] [--log_on_each_node [LOG_ON_EACH_NODE]]
              [--logging_dir LOGGING_DIR] [--logging_strategy {no,steps,epoch}] [--logging_first_step [LOGGING_FIRST_STEP]] [--logging_steps LOGGING_STEPS] [--logging_nan_inf_filter LOGGING_NAN_INF_FILTER]
              [--save_strategy {no,steps,epoch}] [--save_steps SAVE_STEPS] [--save_total_limit SAVE_TOTAL_LIMIT] [--save_on_each_node [SAVE_ON_EACH_NODE]] [--no_cuda [NO_CUDA]] [--seed SEED] [--fp16 [FP16]]
              [--fp16_opt_level FP16_OPT_LEVEL] [--fp16_backend {auto,amp,apex}] [--fp16_full_eval [FP16_FULL_EVAL]] [--local_rank LOCAL_RANK] [--xpu_backend {mpi,ccl}] [--tpu_num_cores TPU_NUM_CORES]
              [--tpu_metrics_debug [TPU_METRICS_DEBUG]] [--debug DEBUG] [--dataloader_drop_last [DATALOADER_DROP_LAST]] [--eval_steps EVAL_STEPS] [--dataloader_num_workers DATALOADER_NUM_WORKERS]
              [--past_index PAST_INDEX] [--run_name RUN_NAME] [--disable_tqdm DISABLE_TQDM] [--no_remove_unused_columns] [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
              [--label_names LABEL_NAMES [LABEL_NAMES ...]] [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]] [--metric_for_best_model METRIC_FOR_BEST_MODEL] [--greater_is_better GREATER_IS_BETTER]
              [--ignore_data_skip [IGNORE_DATA_SKIP]] [--sharded_ddp SHARDED_DDP] [--deepspeed DEEPSPEED] [--label_smoothing_factor LABEL_SMOOTHING_FACTOR] [--adafactor [ADAFACTOR]]
              [--group_by_length [GROUP_BY_LENGTH]] [--length_column_name LENGTH_COLUMN_NAME] [--report_to REPORT_TO [REPORT_TO ...]] [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
              [--no_dataloader_pin_memory] [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]] [--no_skip_memory_metrics] [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
              [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]] [--push_to_hub [PUSH_TO_HUB]] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--hub_model_id HUB_MODEL_ID]
              [--hub_strategy {end,every_save,checkpoint,all_checkpoints}] [--hub_token HUB_TOKEN] [--gradient_checkpointing [GRADIENT_CHECKPOINTING]] [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID]
              [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION] [--push_to_hub_token PUSH_TO_HUB_TOKEN] [--mp_parameters MP_PARAMETERS] [--n_best_size N_BEST_SIZ