
python -m torch.distributed.run \
--nproc_per_node 8 --nnodes 1 --master_port 44144 \
run_pretrain_binary.py \
--dataset_name /path/to/pretraining/data \
--model_name_or_path bert-base-uncased \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--learning_rate 2e-4 \
--max_train_steps 500000 \
--num_warmup_steps 5000 \
--output_dir ./pretrain_output \
--max_seq_length 128 \
--checkpointing_steps 50000 \
--with_tracking \
--report_to wandb 
