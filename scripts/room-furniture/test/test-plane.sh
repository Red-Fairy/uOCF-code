DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/chairs/test-2-4obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room-furniture' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --n_samp 256 --input_size 128 --render_size 32 --frustum_size 128 \
    --model 'uocf_eval' \
    --num_slots 5 --attn_iter 6 --nss_scale 7 --fg_object_size 3 \
    --fixed_locality \
    --attn_momentum 0.5  \
    --exp_id 'stage2-multiobj-plane' --epoch 120 \
    --vis_attn --vis_mask --remove_duplicate \
    --testset_name 'standard' \


