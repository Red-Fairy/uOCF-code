DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-all-test'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 54 --n_img_each_scene 2  \
    --checkpoints_dir 'checkpoints' --name 'kitchen-shiny' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --n_samp 256 --input_size 128 --render_size 32 --frustum_size 128 \
    --model 'uocf_eval' \
    --num_slots 5 --attn_iter 6 --nss_scale 7 --fg_object_size 3 \
    --fixed_locality \
    --attn_momentum 0.5  \
    --exp_id 'stage2-multiobj-plane' --epoch 750 \
    --vis_attn --vis_render_mask --recon_only --remove_duplicate \
    --testset_name 'standard' \


