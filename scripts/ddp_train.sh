cd /home/zyr/RL-UIE/conditional-flow-matching

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_uie.py \
    --parallel True \
    --batch_size 16 \
    --lr 2e-4 \
    --total_steps 400001 \
    --val_step 5000 \
    --nfe 20
