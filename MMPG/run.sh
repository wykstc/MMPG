# set dataset and batch size
dataset='fold'
dataset_path='/Protein/'
batch_size=32
eval_batch_size=32

# set training configuration
num_epochs=400
num_workers=8
lr=0.01
seed=42
device=0
checkpoint_dir="./checkpoints/"

python train.py --device $device --dataset $dataset --dataset_path $dataset_path --batch_size $batch_size --eval_batch_size $eval_batch_size --lr ${lr} --num_workers ${num_workers} --num_epochs ${num_epochs} --save_checkpoint --checkpoint_dir ${checkpoint_dir} --seed $seed
