from itertools import chain

import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection, pixel2world
from .model_general import MultiDINOStackEncoder, SlotAttentionTransformer, DecoderIPE
from .utils import *
from util.util import AverageMeter
from sklearn.metrics import adjusted_rand_score
import lpips
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr
import numpy as np

class uocfEvalModel(BaseModel):

	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new model-specific options and rewrite default values for existing options.

		Parameters:
			parser -- the option parser
			is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
		parser.add_argument('--shape_dim', type=int, default=48, help='Dimension of individual z latent per slot')
		parser.add_argument('--color_dim', type=int, default=48, help='Dimension of individual z latent per slot texture')
		parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
		parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
		parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
		parser.add_argument('--obj_scale', type=float, default=6, help='slot-centric locality constraint')
		parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
		parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
		parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
		parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
		parser.add_argument('--input_size', type=int, default=64)
		parser.add_argument('--frustum_size', type=int, default=128, help='Size of rendered images')
		parser.add_argument('--near_plane', type=float, default=6)
		parser.add_argument('--far_plane', type=float, default=20)
		parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
		parser.add_argument('--fg_in_world', action='store_true', help='foreground objects are in world space')

		parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
							dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

		parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

		return parser

	def __init__(self, opt):
		"""Initialize this model class.

		Parameters:
			opt -- training/test options

		A few things can be done here.
		- (required) call the initialization function of BaseModel
		- define loss function, visualization images, model names, and optimizers
		"""
		BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
		self.loss_names = ['psnr', 'ssim', 'lpips'] if not self.opt.no_loss else []
		if not opt.recon_only and not opt.video:
			self.loss_names += ['ari', 'fgari', 'nvari']
		if opt.show_recon_stats and not opt.no_loss:
			self.loss_names += ['recon_psnr', 'recon_ssim', 'recon_lpips']
		n = opt.n_img_each_scene
		self.set_visual_names()
		self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
		render_size = (opt.render_size, opt.render_size)
		frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
		self.intrinsics = np.loadtxt(os.path.join(opt.dataroot, 'camera_intrinsics_ratio.txt')) if opt.load_intrinsics else None
		self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
									 frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size, intrinsics=self.intrinsics)

		z_dim = opt.color_dim + opt.shape_dim
		self.num_slots = opt.num_slots

		self.pretrained_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device).eval()
		dino_dim = 768
		self.netEncoder = MultiDINOStackEncoder(shape_dim=opt.shape_dim, color_dim=opt.color_dim, input_dim=dino_dim, 
						n_feat_layer=opt.n_feat_layers, kernel_size=opt.enc_kernel_size, mode=opt.enc_mode)

		self.netSlotAttention = SlotAttentionTransformer(num_slots=opt.num_slots, in_dim=opt.shape_dim, 
					slot_dim=opt.shape_dim, color_dim=opt.color_dim, momentum=opt.attn_momentum, pos_init=opt.pos_init,
					learnable_pos=not opt.no_learnable_pos, iters=opt.attn_iter, depth_scale_pred=self.opt.depth_scale_pred, depth_scale_param=opt.depth_scale_param,
					camera_modulation=opt.camera_modulation, camera_dim=16)
							  
		self.netDecoder = DecoderIPE(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=z_dim, n_layers=opt.n_layer, locality=False,
													locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality,
													mlp_act=opt.dec_mlp_act, density_act=opt.dec_density_act,)

		self.netEncoder = self.netEncoder.to(self.device)
		self.netSlotAttention = self.netSlotAttention.to(self.device)
		self.netDecoder = self.netDecoder.to(self.device)
		
		self.L2_loss = torch.nn.MSELoss()
		self.LPIPS_loss = lpips.LPIPS().to(self.device)

		assert self.opt.fixed_locality

	def set_visual_names(self):
		n = self.opt.n_img_each_scene
		n_slot = self.opt.num_slots
		self.visual_names =	['gt_novel_view{}'.format(i+1) for i in range(n-1)] + \
							['x_rec{}'.format(i) for i in range(n)] + \
							['input_image'] + \
							['slot{}_view{}_unmasked'.format(k, i) for k in range(n_slot) for i in range(n)] + \
							['slot{}_view{}'.format(k, i) for k in range(n_slot) for i in range(n)]

		if self.opt.vis_mask:
			self.visual_names += ['mask_gt{}'.format(i) for i in range(n)]
		
		if self.opt.vis_render_mask or self.opt.vis_mask:
			self.visual_names += ['mask_rec{}'.format(i) for i in range(n)]

		if self.opt.vis_attn:
			self.visual_names += ['slot{}_attn'.format(k) for k in range(n_slot)]

		if self.opt.vis_disparity:
			self.visual_names += ['disparity_{}'.format(i) for i in range(n)]

		if self.opt.vis_disparity or self.opt.vis_render_disparity:
			self.visual_names += ['disparity_rec{}'.format(i) for i in range(n)]

	def setup(self, opt):
		"""Load and print networks; create schedulers
		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		if self.isTrain:
			self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
		if not self.isTrain or opt.continue_train:
			load_suffix = 'iter_{}'.format(opt.load_iter) if opt.load_iter > 0 else opt.epoch
			self.load_networks(load_suffix)
		self.print_networks(opt.verbose)

	def set_input(self, input):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input: a dictionary that contains the data itself and its metadata information.
		"""
		self.x = input['img_data'].to(self.device)
		self.x_large = input['img_data_large'].to(self.device)
		self.cam2world = input['cam2world'].to(self.device)
		self.image_paths = input['paths']
		if 'intrinsics' in input:
			self.intrinsics = input['intrinsics'][0].to(self.device).squeeze(0) # overwrite the default intrinsics

		if not self.opt.recon_only and not self.opt.video:
			self.gt_masks = input['masks']
			self.mask_idx = input['mask_idx']
			self.fg_idx = input['fg_idx']
			self.obj_idxs = input['obj_idxs']  # (K+1)x1xHxW
			self.masks_fg = input['obj_idxs_fg'].to(self.device)  # Kx1xHxW

		if self.opt.vis_disparity:
			self.disparity = input['depth'].to(self.device)

	def encode(self, idx=0, return_global_feature=False):
		"""Encode the input image into a feature map.
		Parameters:
			idx: idx of the image to be encoded (typically 0, if position loss is used, we may use 1)
		Returns:
			feat_shape, feat_color (BxHxWxC), (BxHxWxC)
			class_token (BxC)
		"""
		feature_maps, class_tokens = [], []
		with torch.no_grad(): # B*C*H*W
			outputs = self.pretrained_encoder.get_intermediate_layers(self.x_large[idx:idx+1], n=self.opt.n_feat_layers, reshape=True, return_class_token=True)
		
		for feature_map, class_token in outputs:
			feature_maps.append(feature_map)
			if return_global_feature:
				class_tokens.append(class_token)
			else:
				class_tokens.append(None)

		feature_map_shape, feature_map_color, feature_global = self.netEncoder(feature_maps, feature_maps[-1], class_tokens[-1])  # Bxshape_dimxHxW, Bxcolor_dimxHxW

		feat_shape = feature_map_shape.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC
		feat_color = feature_map_color.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC

		return feat_shape, feat_color, feature_global

	def forward(self, epoch=0):
		"""Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
		dev = self.x[0:1].device
		cam2world_viewer = self.cam2world[0]
		nss2cam0 = self.cam2world[0:1].inverse()
		if self.opt.fixed_locality: # divide the translation part by self.opt.nss_scale
			nss2cam0 = torch.cat([torch.cat([nss2cam0[:, :3, :3], nss2cam0[:, :3, 3:4]/self.opt.nss_scale], dim=2), 
									nss2cam0[:, 3:4, :]], dim=1) # 1*4*4

		# Encoding images
		feat_shape, feat_color, _ = self.encode(0)

		# calculate camera cond (R, T, fx, fy, cx, cy) , 1*camera_dim, camera_dim=5, assert camera_normalized
		camExt = torch.cat([cam2world_viewer[:3, :3].flatten(), cam2world_viewer[:3, 3:4].flatten()/self.opt.nss_scale], dim=0)
		camInt = torch.Tensor([self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]]).to(dev) \
			if self.intrinsics is not None else torch.Tensor([350./320., 350./240., 0., 0.]).to(dev)
		camera_modulation = torch.cat([camExt, camInt], dim=0).unsqueeze(0)

		# transformer attention
		z_slots, attn, fg_slot_position, fg_depth_scale = self.netSlotAttention(feat_shape, feat_color=feat_color, camera_modulation=camera_modulation, 
														  remove_duplicate=self.opt.remove_duplicate)  # 1xKxC, 1xKxN, 1xKx2, 1xKx1
		z_slots, attn, fg_slot_position, fg_depth_scale = z_slots.squeeze(0), attn.squeeze(0), fg_slot_position.squeeze(0), fg_depth_scale.squeeze(0)  # KxC, KxN, Kx2, Kx1

		K = z_slots.shape[0] # num_slots - n_remove
		self.num_slots = K

		cam2world = self.cam2world
		N = cam2world.shape[0]

		if not self.opt.scaled_depth:
			fg_slot_nss_position = pixel2world(fg_slot_position, cam2world_viewer, intrinsics=self.intrinsics, nss_scale=self.opt.nss_scale)  # Kx3
		else: 
			depth_scale = self.opt.depth_scale if self.opt.depth_scale is not None else torch.norm(cam2world_viewer[:3, 3:4])
			slot_depth = torch.ones_like(fg_slot_position[:, 0:1]).to(self.x.device) * depth_scale * fg_depth_scale
			fg_slot_nss_position = pixel2world(fg_slot_position, cam2world_viewer, intrinsics=self.intrinsics, 
													nss_scale=self.opt.nss_scale, depth=slot_depth) # Kx3

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size
		(mean, var), z_vals, ray_dir = self.projection.sample_along_rays(cam2world, partitioned=True, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
		# scale**2x(NxHxW)xDx3, scale**2x(NxHxW)xDx3x3, scale**2x(NxHxW)xD, scale**2x(NxHxW)x3
		x = self.x
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		if self.opt.vis_disparity or self.opt.vis_render_disparity:
			disparity_rec = torch.zeros([N, 1, H, W], device=dev)

		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			# print(z_slots.shape, sampling_coor_bg_.shape, sampling_coor_fg_.shape, nss2cam0.shape, fg_slot_nss_position.shape)
			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder.forward(mean_, var_, z_slots, nss2cam0, fg_slot_nss_position, 
								 fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			masked_raws[:, :, h::scale, w::scale, ...] = masked_raws_
			unmasked_raws[:, :, h::scale, w::scale, ...] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_
			if self.opt.vis_disparity or self.opt.vis_render_disparity:
				disparity_rec_ = 1 / depth_map_.view(N, 1, H_, W_)
				disparity_rec[..., h::scale, w::scale] = disparity_rec_

		if self.opt.frustum_size != self.opt.load_size:
			x = F.interpolate(x, size=[H, W], mode='bilinear')

		if not self.opt.no_loss and not self.opt.video:
			x_recon_novel, x_novel = x_recon[1:], x[1:]
			self.loss_recon = self.L2_loss(x_recon_novel, x_novel)
			self.loss_lpips = self.LPIPS_loss(x_recon_novel, x_novel).mean()
			self.loss_psnr = compute_psnr(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)
			self.loss_ssim = compute_ssim(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)

		if not self.opt.no_loss and self.opt.show_recon_stats:
			x_recon_ori, x_ori = x_recon[0:1], x[0:1]
			self.loss_recon_psnr = compute_psnr(x_recon_ori/2+0.5, x_ori/2+0.5, data_range=1.)
			self.loss_recon_ssim = compute_ssim(x_recon_ori/2+0.5, x_ori/2+0.5, data_range=1.)
			self.loss_recon_lpips = self.LPIPS_loss(x_recon_ori, x_ori).mean()

		with torch.no_grad():
			attn = attn.detach().cpu()  # KxN
			H_, W_ = feat_shape.shape[1:3]
			attn = attn.view(self.num_slots, 1, H_, W_)
			# if H_ != H:
			# 	attn = F.interpolate(attn, size=[H, W], mode='bilinear')
			setattr(self, 'attn', attn)

			for i in range(self.opt.n_img_each_scene):
				setattr(self, 'x_rec{}'.format(i), x_recon[i])
				if i == 0:
					setattr(self, 'input_image', x[i])
				else:
					setattr(self, 'gt_novel_view{}'.format(i), x[i])
				if self.opt.vis_disparity: # normalize disparity to [0, 1]
					setattr(self, 'disparity_{}'.format(i), (self.disparity[i] - self.disparity[i].min()) / (self.disparity[i].max() - self.disparity[i].min()))
				if self.opt.vis_render_disparity or self.opt.vis_disparity:
					setattr(self, 'disparity_rec{}'.format(i), (disparity_rec[i] - disparity_rec[i].min()) / (disparity_rec[i].max() - disparity_rec[i].min()))
			setattr(self, 'masked_raws', masked_raws.detach())
			setattr(self, 'unmasked_raws', unmasked_raws.detach())
			setattr(self, 'fg_slot_image_position', fg_slot_position.detach())
			setattr(self, 'fg_slot_nss_position', fg_slot_nss_position.detach())
			setattr(self, 'z_slots', z_slots.detach())

	def visual_cam2world(self, cam2world):
		'''
		render the scene given the cam2world matrix (1x4x4)
		must be called after forward()
		'''
		dev = self.x[0:1].device
		cam2world = cam2world.to(dev)

		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
		if self.opt.fixed_locality: # divide the translation part by self.opt.nss_scale
			nss2cam0 = torch.cat([torch.cat([nss2cam0[:, :3, :3], nss2cam0[:, :3, 3:4]/self.opt.nss_scale], dim=2), 
									nss2cam0[:, 3:4, :]], dim=1) # 1*4*4
			
		K = self.attn.shape[0]
		N = cam2world.shape[0]

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size
		(mean, var), z_vals, ray_dir = self.projection.sample_along_rays(cam2world, partitioned=True, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
		#  4x(Nx(H/2)x(W/2))xDx3, 4x(Nx(H/2)x(W/2))xDx3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		if self.opt.vis_disparity or self.opt.vis_render_disparity:
			disparity_rec = torch.zeros([N, 1, H, W], device=dev)
		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(mean_, var_, self.z_slots, nss2cam0, 
							self.fg_slot_nss_position, fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			masked_raws[..., h::scale, w::scale, :, :] = masked_raws_
			unmasked_raws[..., h::scale, w::scale, :, :] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True, white_bkgd=self.opt.white_bkgd)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_
			if self.opt.vis_disparity or self.opt.vis_render_disparity:
				disparity_rec_ = 1 / depth_map_.view(N, 1, H_, W_)
				disparity_rec[..., h::scale, w::scale] = disparity_rec_

		with torch.no_grad():
			# for i in range(self.opt.n_img_each_scene):
			for i in range(1):
				setattr(self, 'x_rec{}'.format(i), x_recon[i])
			setattr(self, 'masked_raws', masked_raws.detach())
			setattr(self, 'unmasked_raws', unmasked_raws.detach())
			if self.opt.vis_disparity or self.opt.vis_render_disparity: # normalize disparity to [0, 1]
				setattr(self, 'disparity_rec{}'.format(i), (disparity_rec[i] - disparity_rec[i].min()) / (disparity_rec[i].max() - disparity_rec[i].min()))

   
	def forward_position(self, fg_slot_image_position=None, fg_slot_nss_position=None, z_slots=None):
		assert fg_slot_image_position is None or fg_slot_nss_position is None
		x = self.x
		dev = x.device
		cam2world = self.cam2world
		N = cam2world.shape[0]

		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
		if self.opt.fixed_locality: # divide the translation part by self.opt.nss_scale
			nss2cam0 = torch.cat([torch.cat([nss2cam0[:, :3, :3], nss2cam0[:, :3, 3:4]/self.opt.nss_scale], dim=2), 
									nss2cam0[:, 3:4, :]], dim=1) # 1*4*4
		
		if fg_slot_image_position is not None:
			fg_slot_nss_position = pixel2world(fg_slot_image_position.to(dev), self.cam2world[0], intrinsics=self.intrinsics, nss_scale=self.opt.nss_scale)
		elif fg_slot_nss_position is not None:
			if fg_slot_nss_position.shape[1] == 2:
				fg_slot_nss_position = torch.cat([fg_slot_nss_position, torch.zeros_like(fg_slot_nss_position[:, :1])], dim=1)
			fg_slot_nss_position = fg_slot_nss_position.to(dev)
		else:
			fg_slot_nss_position = self.fg_slot_nss_position.to(dev)

		if z_slots is not None:
			z_slots = z_slots.to(dev)
		else:
			z_slots = self.z_slots.to(dev)

		K = z_slots.shape[0]

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size

		(mean, var), z_vals, ray_dir = self.projection.sample_along_rays(cam2world, partitioned=True, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
		# 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		
		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			# print(z_slots.shape, sampling_coor_bg_.shape, sampling_coor_fg_.shape, nss2cam0.shape, fg_slot_nss_position.shape)
			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(mean_, var_, z_slots, 
								 nss2cam0, fg_slot_nss_position, fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			masked_raws[..., h::scale, w::scale, :, :] = masked_raws_
			unmasked_raws[..., h::scale, w::scale, :, :] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_

		with torch.no_grad():
			for i in range(self.opt.n_img_each_scene):
				setattr(self, 'x_rec{}'.format(i), x_recon[i])
			setattr(self, 'masked_raws', masked_raws.detach())
			setattr(self, 'unmasked_raws', unmasked_raws.detach())
			if fg_slot_image_position is not None:
				setattr(self, 'fg_slot_image_position', fg_slot_image_position.detach())
			setattr(self, 'fg_slot_nss_position', fg_slot_nss_position.detach())

	def compute_visuals(self, cam2world=None):
		with torch.no_grad():
			cam2world = self.cam2world[:self.opt.n_img_each_scene] if cam2world is None else cam2world.to(self.device)
			# n_img_each_scene = cam2world.shape[0]
			_, N, H, W, D, _ = self.masked_raws.shape
			masked_raws = self.masked_raws  # KxNxHxWxDx4
			unmasked_raws = self.unmasked_raws  # KxNxHxWxDx4
			mask_maps = []
			
			for k in range(self.num_slots):
				setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)
			for k in range(self.num_slots, self.opt.num_slots):
				setattr(self, 'slot{}_attn'.format(k), torch.zeros_like(self.attn[0]))

			for k in range(self.num_slots):
				raws = unmasked_raws[k]  # NxDxHxWx4
				_, z_vals, ray_dir = self.projection.sample_along_rays(cam2world, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
				raws = raws.flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
				rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True, mip=True)
				# mask_maps.append(mask_map.view(N, H, W))
				rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
				x_recon = rendered * 2 - 1
				for i in range(N):
					setattr(self, 'slot{}_view{}_unmasked'.format(k, i), x_recon[i])
			for k in range(self.num_slots, self.opt.num_slots):
				for i in range(N):
					setattr(self, 'slot{}_view{}_unmasked'.format(k, i), torch.zeros_like(x_recon[i]))

			for k in range(self.num_slots):
				raws = masked_raws[k]  # NxDxHxWx4
				_, z_vals, ray_dir = self.projection.sample_along_rays(cam2world, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
				raws = raws.flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
				rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True, mip=True)
				mask_maps.append(mask_map.view(N, H, W))
				rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
				x_recon = rendered * 2 - 1
				for i in range(N):
					setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])
			for k in range(self.num_slots, self.opt.num_slots):
				for i in range(N):
					setattr(self, 'slot{}_view{}'.format(k, i), torch.zeros_like(x_recon[i]))

			if self.opt.vis_render_mask and self.opt.recon_only:
				# mask_maps_fg = torch.stack(mask_maps[1:])
				# mask_sum_fg = torch.sum(mask_maps_fg, dim=0, keepdim=True)
				# mask_maps_fgbg = torch.cat([torch.Tensor(mask_maps[0]).unsqueeze(0), mask_sum_fg], dim=0)
				# mask_idx_coarse = mask_maps_fgbg.cpu().argmax(dim=0)  # NxHxW
				# mask_idx_fine = mask_maps_fg.cpu().argmax(dim=0)  # NxHxW
				# mask_idx = mask_idx_coarse * (mask_idx_fine + 1)
				mask_maps = torch.stack(mask_maps)  # KxNxHxW
				mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW
				# define 8 colors from color palette
				color_palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
								 [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0], [0, 0.5, 0.5]]
				colors = torch.cat([torch.tensor([-1., -1., -1.]).view([1, 3]), torch.tensor(color_palette) * 2 - 1], dim=0).to(self.device)  # Kx3
				mask_visuals = colors[mask_idx]  # NxHxWx3
				for i in range(N):
					setattr(self, 'mask_rec{}'.format(i), mask_visuals[i, ...].permute([2, 0, 1]))

			if not self.opt.recon_only and not self.opt.video:
				mask_maps = torch.stack(mask_maps)  # KxNxHxW
				mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW
				predefined_colors = []
				obj_idxs = self.obj_idxs  # Kx1xHxW
				gt_mask0 = self.gt_masks[0]  # 3xHxW
				for k in range(self.num_slots):
					mask_idx_this_slot = mask_idx[0:1] == k  # 1xHxW
					iou_this_slot = []
					for kk in range(self.num_slots):
						try:
							obj_idx = obj_idxs[kk, ...]  # 1xHxW
						except IndexError:
							break
						iou = (obj_idx & mask_idx_this_slot).type(torch.float).sum() / (obj_idx | mask_idx_this_slot).type(torch.float).sum()
						iou_this_slot.append(iou)
					target_obj_number = torch.tensor(iou_this_slot).argmax()
					target_obj_idx = obj_idxs[target_obj_number, ...].squeeze()  # HxW
					obj_first_pixel_pos = target_obj_idx.nonzero()[0]  # 2
					obj_color = gt_mask0[:, obj_first_pixel_pos[0], obj_first_pixel_pos[1]]
					predefined_colors.append(obj_color)
				predefined_colors = torch.stack(predefined_colors).permute([1,0])
				mask_visuals = predefined_colors[:, mask_idx]  # 3xNxHxW

				nvari_meter = AverageMeter()
				for i in range(N):
					setattr(self, 'mask_rec{}'.format(i), mask_visuals[:, i, ...])
					setattr(self, 'mask_gt{}'.format(i), self.gt_masks[i])
					this_mask_idx = mask_idx[i].flatten(start_dim=0)
					gt_mask_idx = self.mask_idx[i]  # HW
					fg_idx = self.fg_idx[i]
					fg_idx_map = fg_idx.view([self.opt.frustum_size, self.opt.frustum_size])[None, ...]
					fg_map = mask_visuals[0:1, i, ...].clone()
					fg_map[fg_idx_map] = -1.
					fg_map[~fg_idx_map] = 1.
					setattr(self, 'bg_map{}'.format(i), fg_map)
					if i == 0:
						ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
						fg_ari = adjusted_rand_score(gt_mask_idx[fg_idx], this_mask_idx[fg_idx])
						self.loss_ari = ari_score
						self.loss_fgari = fg_ari
					else:
						ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
						nvari_meter.update(ari_score)
					self.loss_nvari = nvari_meter.val

	def backward(self):
		pass

	def optimize_parameters(self, ret_grad=False, epoch=0):
		"""Update network weights; it will be called in every training iteration."""
		self.forward(epoch)
		for opm in self.optimizers:
			opm.zero_grad()
		self.backward()
		avg_grads = []
		layers = []
		if ret_grad:
			for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
				if p.grad is not None and "bias" not in n:
					with torch.no_grad():
						layers.append(n)
						avg_grads.append(p.grad.abs().mean().cpu().item())
		for opm in self.optimizers:
			opm.step()
		return layers, avg_grads

	def save_networks(self, surfix):
		"""Save all the networks to the disk.

		Parameters:
			surfix (int or str) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		super().save_networks(surfix)
		for i, opm in enumerate(self.optimizers):
			save_filename = '{}_optimizer_{}.pth'.format(surfix, i)
			save_path = os.path.join(self.save_dir, save_filename)
			torch.save(opm.state_dict(), save_path)

		for i, sch in enumerate(self.schedulers):
			save_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
			save_path = os.path.join(self.save_dir, save_filename)
			torch.save(sch.state_dict(), save_path)

	def load_networks(self, surfix):
		"""Load all the networks from the disk.

		Parameters:
			surfix (int or str) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
		"""
		super().load_networks(surfix)

		if self.isTrain:
			for i, opm in enumerate(self.optimizers):
				load_filename = '{}_optimizer_{}.pth'.format(surfix, i)
				load_path = os.path.join(self.save_dir, load_filename)
				print('loading the optimizer from %s' % load_path)
				state_dict = torch.load(load_path, map_location=str(self.device))
				opm.load_state_dict(state_dict)

			for i, sch in enumerate(self.schedulers):
				load_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
				load_path = os.path.join(self.save_dir, load_filename)
				print('loading the lr scheduler from %s' % load_path)
				state_dict = torch.load(load_path, map_location=str(self.device))
				sch.load_state_dict(state_dict)


if __name__ == '__main__':
	pass