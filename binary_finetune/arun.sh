CUDA_VISIBLE_DEVICES=4 nohup sh run.sh cola 50 binary > cola.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run.sh mnli 6 binary > mnli.out 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run.sh mrpc 20 binary > mrpc.out 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run.sh sst2 10 binary > sst-2.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run.sh stsb 20 binary > sts-b.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run.sh qqp 6 binary > qqp.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run.sh qnli 10 binary > qnli.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run.sh rte 20 binary > rte.out 2>&1 &


