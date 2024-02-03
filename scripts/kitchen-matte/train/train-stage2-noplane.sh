DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-easy/4obj-all-train-0817'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes 735 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'kitchen-matte' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 5 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uocf' --lr 0.00015 \
    --seed 2024 \
    --num_slots 2 --attn_iter 6 --nss_scale 7 --fg_object_size 3 \
    --stratified --fixed_locality \
    --bg_density_loss --depth_supervision --remove_duplicate --remove_duplicate_in 100 \
    --load_pretrain --load_pretrain_path './checkpoints/room-texture/stage1-1obj-noplane' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' --load_epoch 100 --one2four \
    --coarse_epoch 250 --niter 750 --percept_in 50 --no_locality_epoch 150 --dense_sample_epoch 150 \
    --scaled_depth --depth_scale_pred \
    --exp_id 'stage2-multiobj-noplane' \
    
