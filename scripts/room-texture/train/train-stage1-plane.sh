DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/chairs/train-1obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train.py --dataroot $DATAROOT --n_scenes 1296 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'room-texture' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uocf' \
    --seed 2025 \
    --num_slots 2 --attn_iter 6 --nss_scale 7 --fg_object_size 3 \
    --stratified --fixed_locality \
    --coarse_epoch 100 --niter 100 --percept_in 10 --no_locality_epoch 20 \
    --fg_density_loss --bg_density_loss --bg_density_in 10 --collapse_prevent 5000 \
    --exp_id 'stage1-1obj-plane' \

