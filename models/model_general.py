import torch
from torch import nn
from .utils import PositionalEncoding, build_grid

class Encoder(nn.Module):
	def __init__(self, input_nc=3, z_dim=64, bottom=False, double_bottom=False, pos_emb=False):

		super().__init__()

		self.bottom = bottom
		self.double_bottom = double_bottom
		assert double_bottom == False or bottom == True
		print('Bottom for Encoder: ', self.bottom)
		print('Double Bottom for Encoder: ', self.double_bottom)

		input_nc = input_nc + 4 if pos_emb else input_nc
		self.pos_emb = pos_emb

		if self.bottom and self.double_bottom:
			self.enc_down_00 = nn.Sequential(nn.Conv2d(input_nc, z_dim // 2, 3, stride=1, padding=1),
																nn.ReLU(True))
			self.enc_down_01 = nn.Sequential(nn.Conv2d(z_dim // 2, z_dim, 3, stride=2, padding=1),
																nn.ReLU(True))
	
		elif self.bottom:
			self.enc_down_0 = nn.Sequential(nn.Conv2d(input_nc, z_dim, 3, stride=1, padding=1),
											nn.ReLU(True))
			
		self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc, z_dim, 3, stride=2 if bottom else 1, padding=1),
										nn.ReLU(True))
		self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
										nn.ReLU(True))
		self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
										nn.ReLU(True))
		self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
									  nn.ReLU(True),
									  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
									  nn.ReLU(True),
									  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
									#   nn.ReLU(True)
									  )

	def forward(self, x):
		"""
		input:
			x: input image, Bx3xHxW
		output:
			feature_map: BxCxHxW
		"""

		if self.pos_emb:
			W, H = x.shape[3], x.shape[2]
			X = torch.linspace(-1, 1, W)
			Y = torch.linspace(-1, 1, H)
			y1_m, x1_m = torch.meshgrid([Y, X])
			x2_m, y2_m = -x1_m, -y1_m  # Normalized distance in the four direction
			pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).unsqueeze(0)  # 1x4xHxW
			x_ = torch.cat([x, pixel_emb], dim=1)
		else:
			x_ = x

		if self.bottom and self.double_bottom:
			x_down_00 = self.enc_down_00(x_)
			x_down_01 = self.enc_down_01(x_down_00)
			x_down_1 = self.enc_down_1(x_down_01)
		elif self.bottom:
			x_down_0 = self.enc_down_0(x_)
			x_down_1 = self.enc_down_1(x_down_0)
		else:
			x_down_1 = self.enc_down_1(x_)
		x_down_2 = self.enc_down_2(x_down_1)
		x_down_3 = self.enc_down_3(x_down_2)
		x_up_3 = self.enc_up_3(x_down_3)
		x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
		feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
		
		feature_map = feature_map
		return feature_map

class MultiDINOStackEncoder(nn.Module):
	def __init__(self, n_feat_layer=1, shape_dim=64, color_dim=64, input_dim=256, hidden_dim=64, 
			  	DINO_dim=768, kernel_size=3, mode='sum', global_bg_feature=False):
		super().__init__()

		self.mode = mode
		if mode == 'sum':
			self.shallow_encoders = nn.ModuleList([nn.Sequential(
												nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
												nn.ReLU(True),
												) for _ in range(n_feat_layer)])
		elif mode == 'stack':
			self.shallow_encoders = nn.ModuleList([nn.Sequential(
											nn.Conv2d(input_dim, hidden_dim // n_feat_layer, kernel_size=kernel_size, stride=1, padding='same'),
											nn.ReLU(True),
											) for _ in range(n_feat_layer)])
		else:
			assert False, 'mode not supported'
						
		self.combine = nn.Conv2d(hidden_dim, shape_dim, kernel_size=kernel_size, stride=1, padding='same')
		
		self.stack_encoder = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding='same'),
											nn.ReLU(True),
											nn.Conv2d(hidden_dim, color_dim, kernel_size=kernel_size, stride=1, padding='same'))
		if global_bg_feature:
			self.bg_feat = nn.Linear(DINO_dim, color_dim)

	def forward(self, input_feats_shape, input_feat_color, bg_feat=None):
		'''
		input:
			input_feats_shape: list of (B, input_dim, 64, 64)
			input_feat_color: (B, input_dim, 64, 64)
		output:
			spatial feature (B, shape_dim, 64, 64)
		'''
		feats_shape = [shallow_encoder(input_feat) for shallow_encoder, input_feat in zip(self.shallow_encoders, input_feats_shape)]
		if self.mode == 'sum':
			feat_shape = torch.sum(torch.stack(feats_shape), dim=0) / len(feats_shape)
		elif self.mode == 'stack':
			feat_shape = torch.cat(feats_shape, dim=1)

		feat_shape = self.combine(feat_shape)
		feat_color = self.stack_encoder(input_feat_color)

		if bg_feat is not None:
			return feat_shape, feat_color, self.bg_feat(bg_feat)
		else:
			return feat_shape, feat_color, None

class InputPosEmbedding(nn.Module):
	def __init__(self, in_dim):
		super().__init__()
		self.point_conv = nn.Conv2d(in_dim+4, in_dim, 1, bias=False)
		# init as eye matrix
		self.point_conv.weight.data.zero_()
		for i in range(in_dim):
			self.point_conv.weight.data[i, i, 0, 0] = 1

	def forward(self, x): # x: B*H*W*C
		x = x.permute(0, 3, 1, 2) # B*C*H*W
		W, H = x.shape[3], x.shape[2]
		X = torch.linspace(-1, 1, W)
		Y = torch.linspace(-1, 1, H)
		y1_m, x1_m = torch.meshgrid([Y, X])
		x2_m, y2_m = -x1_m, -y1_m  # Normalized distance in the four direction
		pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).unsqueeze(0)  # 1x4xHxW
		x_ = torch.cat([x, pixel_emb], dim=1)
		# print(x_.shape)
		return self.point_conv(x_).permute(0, 2, 3, 1) # B*H*W*C

class EncoderPosEmbedding(nn.Module):
	def __init__(self, dim, slot_dim, hidden_dim=128):
		super().__init__()
		self.grid_embed = nn.Linear(4, dim, bias=True)
		self.input_to_k_fg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_fg = nn.Linear(dim, dim, bias=False)

		self.input_to_k_bg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_bg = nn.Linear(dim, dim, bias=False)

		self.MLP_fg = nn.Linear(dim, slot_dim, bias=False)
		self.MLP_bg = nn.Linear(dim, slot_dim, bias=False)
		
	def apply_rel_position_scale(self, grid, position):
		"""
		grid: (1, h, w, 2)
		position (batch, number_slots, 2)
		"""
		b, n, _ = position.shape
		h, w = grid.shape[1:3]
		grid = grid.view(1, 1, h, w, 2)
		grid = grid.repeat(b, n, 1, 1, 1)
		position = position.view(b, n, 1, 1, 2)
		
		return grid - position # (b, n, h, w, 2)

	def forward(self, x, h, w, position_latent=None):

		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		if position_latent is not None:
			rel_grid = self.apply_rel_position_scale(grid, position_latent)
		else:
			rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)

		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, n_slot-1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, n_slot-1, h*w, d)

		k, v = self.input_to_k_fg(x).unsqueeze(1), self.input_to_v_fg(x).unsqueeze(1)
		k, v = k + grid_embed, v + grid_embed
		k, v = self.MLP_fg(k), self.MLP_fg(v)

		return k, v # (b, n, h*w, d)

	def forward_bg(self, x, h, w):
		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)
		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, 1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, 1, h*w, d)
		
		k_bg, v_bg = self.input_to_k_bg(x).unsqueeze(1), self.input_to_v_bg(x).unsqueeze(1) # (b, 1, h*w, d)
		k_bg, v_bg = self.MLP_bg(k_bg + grid_embed), self.MLP_bg(v_bg + grid_embed)

		return k_bg, v_bg # (b, 1, h*w, d)

class AdaLN(nn.Module):
	def __init__(self, cond_dim, input_dim, condition=False):
		super().__init__()
		self.norm = nn.LayerNorm(input_dim)

		if condition:
			self.cond_fc = nn.Sequential(nn.Linear(cond_dim, input_dim*2, bias=True), nn.Tanh())
			self.cond_fc[0].weight.data.zero_()
			self.cond_fc[0].bias.data.zero_()
		else:
			self.cond_fc = None

	def forward(self, x, cond=None):
		"""
		x: (B, input_dim)
		cond: (B, cond_dim)
		return: (B, input_dim), input after AdaLN
		"""
		x = self.norm(x)

		if self.cond_fc is None or cond is None:
			return x
		else:
			cond_gamma, cond_beta = self.cond_fc(cond).chunk(2, dim=-1)
			return x * (1 + cond_gamma) + cond_beta

class SlotAttentionTransformer(nn.Module):
	def __init__(self, num_slots, in_dim=64, slot_dim=64, color_dim=8, iters=4, eps=1e-8,
		  learnable_pos=True, n_feats=64*64,
		  momentum=0.5, pos_init='learnable', depth_scale_pred=False, depth_scale_param=2,
		  camera_dim=5, camera_modulation=False, ):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5
		self.pos_momentum = momentum
		self.pos_init = pos_init

		if self.pos_init == 'learnable':
			self.fg_position = nn.Parameter(torch.rand(1, num_slots-1, 2) * 1.5 - 0.75)
		
		self.slots_init_fg = nn.Parameter((torch.randn(1, num_slots-1, slot_dim)))
		self.slots_init_bg = nn.Parameter((torch.randn(1, 1, slot_dim)))

		self.learnable_pos = learnable_pos
		if self.learnable_pos:
			self.attn_to_pos_bias = nn.Sequential(nn.Linear(n_feats, 2), nn.Tanh()) # range (-1, 1)
			self.attn_to_pos_bias[0].weight.data.zero_()
			self.attn_to_pos_bias[0].bias.data.zero_()
		
		self.depth_scale_pred = depth_scale_pred
		if depth_scale_pred:
			self.scale_bias = nn.Sequential(nn.Linear(2+camera_dim+slot_dim+color_dim, 1), nn.Tanh()) # range (-1, 1)
			self.scale_bias[0].weight.data.zero_()
			self.scale_bias[0].bias.data.zero_()
			self.depth_scale_param = depth_scale_param

		self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)

		self.to_q_fg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.to_q_fg =  nn.Linear(slot_dim, slot_dim, bias=False)
		self.to_q_bg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.to_q_bg =  nn.Linear(slot_dim, slot_dim, bias=False)

		self.norm_feat = nn.LayerNorm(in_dim)
		if color_dim != 0:
			self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

		self.mlp_fg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.mlp_fg = nn.Sequential(nn.Linear(slot_dim, slot_dim), 
							  nn.GELU(), nn.Linear(slot_dim, slot_dim))
		self.mlp_bg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.mlp_bg = nn.Sequential(nn.Linear(slot_dim, slot_dim),
							  nn.GELU(), nn.Linear(slot_dim, slot_dim))


	def forward(self, feat, camera_modulation, feat_color=None, num_slots=None, remove_duplicate=False):
		"""
		input:
			feat: visual feature with position information, BxHxWxC
			feat_color: texture feature with position information, BxHxWxC'
			output: slots: BxKxC, attn: BxKxN
		"""
		B, H, W, _ = feat.shape
		N = H * W
		feat = feat.flatten(1, 2) # (B, N, C)

		K = num_slots if num_slots is not None else self.num_slots
		
		if self.pos_init == 'learnable':
			fg_position = self.fg_position.expand(B, -1, -1).to(feat.device)
		elif self.pos_init == 'random':
			fg_position = torch.rand(B, K-1, 2, device=feat.device) * 1.8 - 0.9 # (B, K-1, 2)
		elif self.pos_init == 'zero':
			fg_position = torch.zeros(B, K-1, 2, device=feat.device)
		else:
			assert False

		slot_fg = self.slots_init_fg.expand(B, -1, -1) # (B, K-1, C)
		slot_bg = self.slots_init_bg.expand(B, 1, -1) # (B, 1, C)
		
		feat = self.norm_feat(feat)

		k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2) # (1,N,2)

		for it in range(self.iters):
			n_remove = 0
			if remove_duplicate and it == self.iters - 1:
				remove_idx = []
				with torch.no_grad():
					# calculate similarity matrix between slots
					slot_fg_norm = slot_fg / slot_fg.norm(dim=-1, keepdim=True) # (B,K-1,C)
					similarity_matrix = torch.matmul(slot_fg_norm, slot_fg_norm.transpose(1, 2)) # (B,K-1,K-1)
					pos_diff = fg_position.unsqueeze(2) - fg_position.unsqueeze(1) # (B,K-1,1,2) - (B,1,K-1,2) -> (B,K-1,K-1,2)
					pos_diff_norm = pos_diff.norm(dim=-1) # (B,K-1,K-1)
					for i in range(K-1): # if similarity_matrix[i,j] > 0.75 and pos_diff_norm[i,j] < 0.1, then remove slot j
						if i in remove_idx:
							continue
						for j in range(i+1, K-1):
							if similarity_matrix[:, i, j] > 0.75 and pos_diff_norm[:, i, j] < 0.15 and j not in remove_idx:
								remove_idx.append(j)
					# shift the index (remove the duplicate)
					remove_idx = sorted(remove_idx)
					shuffle_idx = [i for i in range(K-1) if i not in remove_idx]
					# shuffle_idx.extend(remove_idx)
					slot_fg = slot_fg[:, shuffle_idx]
					fg_position = fg_position[:, shuffle_idx]
					n_remove = len(remove_idx)

			q_fg = self.to_q_fg(self.to_q_fg_AdaLN(slot_fg, camera_modulation)) # (B,K-1,C)
			q_bg = self.to_q_bg(self.to_q_bg_AdaLN(slot_bg, camera_modulation)) # (B,1,C)
		
			attn = torch.empty(B, K-n_remove, N, device=feat.device)
			
			k, v = self.to_kv(feat, H, W, fg_position) # (B,K-1,N,C), (B,K-1,N,C)
			
			for i in range(K-n_remove):
				if i != 0:
					k_i = k[:, i-1] # (B,N,C)
					slot_qi = q_fg[:, i-1] # (B,C)
					attn[:, i] = torch.einsum('bd,bnd->bn', slot_qi, k_i) * self.scale
				else:
					attn[:, i] = torch.einsum('bd,bnd->bn', q_bg.squeeze(1), k_bg.squeeze(1)) * self.scale
			
			attn = attn.softmax(dim=1) + self.eps  # BxKxN
			attn_fg, attn_bg = attn[:, 1:, :], attn[:, 0:1, :]  # Bx(K-1)xN, Bx1xN
			attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN
			attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
			
			# momentum update slot position
			# fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
			fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) * (1 - self.pos_momentum) + fg_position * self.pos_momentum

			if it != self.iters - 1:
				updates_fg = torch.empty(B, K-1-n_remove, self.slot_dim, device=k.device) # (B,K-1,C)
				for i in range(K-1-n_remove):
					v_i = v[:, i] # (B,N,C)
					attn_i = attn_weights_fg[:, i] # (B,N)
					updates_fg[:, i] = torch.einsum('bn,bnd->bd', attn_i, v_i)

				updates_bg = torch.einsum('bn,bnd->bd',attn_weights_bg.squeeze(1), v_bg.squeeze(1)) # (B,N,C) * (B,N) -> (B,C)
				updates_bg = updates_bg.unsqueeze(1) # (B,1,C)

				slot_bg = slot_bg + updates_bg
				slot_fg = slot_fg + updates_fg

				slot_bg = slot_bg + self.mlp_bg(self.mlp_bg_AdaLN(slot_bg, camera_modulation))
				slot_fg = slot_fg + self.mlp_fg(self.mlp_fg_AdaLN(slot_fg, camera_modulation))

			else:
				if self.learnable_pos: # add a bias term
					fg_position = fg_position + self.attn_to_pos_bias(attn_weights_fg) * 0.1 # (B,K-1,2)
					fg_position = fg_position.clamp(-1, 1) # (B,K-1,2)
					
				if feat_color is not None:
					# calculate slot color feature
					feat_color = self.norm_feat_color(feat_color)
					feat_color = feat_color.flatten(1, 2) # (B,N,C')
					slot_fg_color = torch.einsum('bkn,bnd->bkd', attn_weights_fg, feat_color) # (B,K-1,N) * (B,N,C') -> (B,K-1,C')
					slot_bg_color = torch.einsum('bn,bnd->bd', attn_weights_bg.squeeze(1), feat_color).unsqueeze(1) # (B,N) * (B,N,C') -> (B,C'), (B,1,C')

		if feat_color is not None:
			slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # (B,K-1,C+C')
			slot_bg = torch.cat([slot_bg, slot_bg_color], dim=-1) # (B,1,C+C')

		if self.depth_scale_pred:
			fg_depth_scale = self.scale_bias(torch.cat([fg_position, camera_modulation.unsqueeze(1).repeat(1, fg_position.shape[1], 1), slot_fg], dim=-1)) / self.depth_scale_param + 1 # (B,K-1,1)
		else:
			fg_depth_scale = torch.ones(B, K-1-n_remove, 1, device=feat.device)
			
		slots = torch.cat([slot_bg, slot_fg], dim=1) # (B,K,C+C')
		
		return slots, attn, fg_position, fg_depth_scale

class DecoderIPE(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, 
		  			locality_ratio=4/7, fixed_locality=False,
					mlp_act='relu', density_act='relu'):
		"""
		freq: raised frequency
		input_dim: pos emb dim + slot dim
		z_dim: network latent dim
		n_layers: #layers before/after skip connection.mlp_act
		locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
		locality_ratio: if locality, what value is the boundary to clamp?
		fixed_locality: if True, compute locality in world space instead of in transformed view space
		"""
		super().__init__()
		super().__init__()
		self.n_freq = n_freq
		self.locality = locality
		self.locality_ratio = locality_ratio
		self.fixed_locality = fixed_locality
		assert self.fixed_locality == True
		self.out_ch = 4
		self.z_dim = z_dim
		activation_mlp = self._build_activation(mlp_act)
		before_skip = [nn.Linear(input_dim, z_dim), activation_mlp()]
		after_skip = [nn.Linear(z_dim+input_dim, z_dim), activation_mlp()]
		for i in range(n_layers-1):
			before_skip.append(nn.Linear(z_dim, z_dim))
			before_skip.append(activation_mlp())
			after_skip.append(nn.Linear(z_dim, z_dim))
			after_skip.append(activation_mlp())
		self.f_before = nn.Sequential(*before_skip)
		self.f_after = nn.Sequential(*after_skip)
		self.f_after_latent = nn.Linear(z_dim, z_dim)
		self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
		self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim//4),
									 activation_mlp(),
									 nn.Linear(z_dim//4, 3))
		before_skip = [nn.Linear(input_dim, z_dim), activation_mlp()]
		after_skip = [nn.Linear(z_dim + input_dim, z_dim), activation_mlp()]
		for i in range(n_layers - 1):
			before_skip.append(nn.Linear(z_dim, z_dim))
			before_skip.append(activation_mlp())
			after_skip.append(nn.Linear(z_dim, z_dim))
			after_skip.append(activation_mlp())
		after_skip.append(nn.Linear(z_dim, self.out_ch))
		self.b_before = nn.Sequential(*before_skip)
		self.b_after = nn.Sequential(*after_skip)

		self.pos_enc = PositionalEncoding(max_deg=n_freq)

		if density_act == 'relu':
			self.density_act = torch.relu
		elif density_act == 'softplus':
			self.density_act = torch.nn.functional.softplus
		else:
			assert False, 'density_act should be relu or softplus'

	def processQueries(self, mean, var, fg_transform, fg_slot_position, z_fg, z_bg, fg_object_size=None):
		'''
		Process the query points and the slot features
		1. If self.fg_object_size is not None, do:
			Remove the query point that is too far away from the slot center, 
			the bouding box is defined as a cube with side length 2 * self.fg_object_size
			for the points outside the bounding box, keep only keep_ratio of them
			store the new sampling_coor_fg and the indices of the remaining points
		2. Do the pos emb by Fourier
		3. Concatenate the pos emb and the slot features
		4. If self.fg_object_size is not None, return the new sampling_coor_fg and their indices

		input: 	mean: PxDx3
				var: PxDx3
				fg_transform: 1x4x4
				fg_slot_position: (K-1)x3
				z_fg: (K-1)xC
				z_bg: 1xC
				ssize: supervision size (64)
				mask_ratio: frequency mask ratio to the pos emb
				rel_pos: use relative position to fg_slot_position or not
				bg_rotate: whether to rotate the background points to the camera coordinate
		return: input_fg: M * (60 + C) (M is the number of query points inside bbox), C is the slot feature dim, and 60 means increased-freq feat dim
				input_bg: Px(60+C)
				idx: M (indices of the query points inside bbox)
		'''
		P, D = mean.shape[0], mean.shape[1]
		K = z_fg.shape[0] + 1

		# only keep the points that inside the cube, ((K-1)*P*D)
		mask_locality = (torch.norm(mean.flatten(0,1), dim=-1) < self.locality_ratio).expand(K-1, -1).flatten(0, 1) if self.locality else torch.ones((K-1)*P*D, device=mean.device).bool()
		# mask_locality = torch.all(torch.abs(mean.flatten(0,1)) < self.locality_ratio, dim=-1).expand(K-1, -1).flatten(0, 1) if self.locality else torch.ones((K-1)*P*D, device=mean.device).bool()
		
		sampling_mean_fg = mean[None, ...].expand(K-1, -1, -1, -1).flatten(1, 2) # (K-1)*(P*D)*3

		sampling_mean_fg = torch.cat([sampling_mean_fg, torch.ones_like(sampling_mean_fg[:, :, 0:1])], dim=-1)  # (K-1)*(P*D)*4
		sampling_mean_fg = torch.matmul(fg_transform[None, ...], sampling_mean_fg[..., None]).squeeze(-1)  # (K-1)*(P*D)*4
		sampling_mean_fg = sampling_mean_fg[:, :, :3]  # (K-1)*(P*D)*3
		
		fg_slot_position = torch.cat([fg_slot_position, torch.ones_like(fg_slot_position[:, 0:1])], dim=-1)  # (K-1)x4
		fg_slot_position = torch.matmul(fg_transform.squeeze(0), fg_slot_position.t()).t() # (K-1)x4
		fg_slot_position = fg_slot_position[:, :3]  # (K-1)x3

		sampling_mean_fg = sampling_mean_fg - fg_slot_position[:, None, :]  # (K-1)x(P*D)x3

		sampling_mean_fg = sampling_mean_fg.view([K-1, P, D, 3]).flatten(0, 1)  # ((K-1)xP)xDx3
		sampling_var_fg = var[None, ...].expand(K-1, -1, -1, -1).flatten(0, 1)  # ((K-1)xP)xDx3

		sampling_mean_bg, sampling_var_bg = mean, var

		# 1. Remove the query points too far away from the slot center
		if fg_object_size is not None:
			sampling_mean_fg_ = sampling_mean_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xPxD)x3
			mask = torch.all(torch.abs(sampling_mean_fg_) < fg_object_size, dim=-1)  # ((K-1)xPxD) --> M
			mask = mask & mask_locality
			if mask.sum() <= 1:
				mask[:2] = True # M == 0 / 1, keep at least two points to avoid error
			idx = mask.nonzero().squeeze()  # Indices of valid points
		else:
			idx = mask_locality.nonzero().squeeze()
			# print('mask ratio: ', 1 - mask_locality.sum().item() / (K-1) / P / D)

		# 2. Compute Fourier position embeddings
		pos_emb_fg = self.pos_enc(sampling_mean_fg, sampling_var_fg)[0]  # ((K-1)xP)xDx(6*n_freq+3)
		pos_emb_bg = self.pos_enc(sampling_mean_bg, sampling_var_bg)[0]  # PxDx(6*n_freq+3)

		pos_emb_fg, pos_emb_bg = pos_emb_fg.flatten(0, 1)[idx], pos_emb_bg.flatten(0, 1)  # Mx(6*n_freq+3), (P*D)x(6*n_freq+3)

		# 3. Concatenate the embeddings with z_fg and z_bg features
		# Assuming z_fg and z_bg are repeated for each query point
		# Also assuming K is the first dimension of z_fg and we need to repeat it for each query point
		
		z_fg = z_fg[:, None, :].expand(-1, P*D, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xPxD)xC
		z_fg = z_fg[idx]  # MxC

		input_fg = torch.cat([pos_emb_fg, z_fg], dim=-1)
		input_bg = torch.cat([pos_emb_bg, z_bg.repeat(P*D, 1)], dim=-1) # (P*D)x(6*n_freq+3+C)

		# 4. Return required tensors
		return input_fg, input_bg, idx, sampling_mean_fg.flatten(0, 1)[idx]

	def forward(self, mean, var, z_slots, fg_transform, fg_slot_position, dens_noise=0., 
		 			fg_object_size=None):
		"""
		1. pos emb by Fourier
		2. for each slot, decode all points from coord and slot feature
		input:
			mean: P*D*3, P = (N*H*W)
			var: P*D*3, P = (N*H*W)
			view_dirs: P*3, P = (N*H*W)
			z_slots: KxC, K: #slots, C: #feat_dim
			z_slots_texture: KxC', K: #slots, C: #texture_dim
			fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0 in nss space,
							otherwise it is 1x3x3 azimuth rotation of nss2cam0 (not used)
			fg_slot_position: (K-1)x3 in nss space
			dens_noise: Noise added to density
			add_blob: add a blob to foreground slots in early iterations to avoid empty slots (gaussian pdf)

			if fg_slot_cam_position is not None, we should first project it world coordinates
			depth: K*1, depth of the slots
		"""
		K, C = z_slots.shape
		P, D = mean.shape[0], mean.shape[1]
		
		z_bg = z_slots[0:1, :]  # 1xC
		z_fg = z_slots[1:, :]  # (K-1)xC

		input_fg, input_bg, idx, fg_means = self.processQueries(mean, var, fg_transform, fg_slot_position, z_fg, z_bg, 
						fg_object_size=fg_object_size)
		
		tmp = self.b_before(input_bg)
		bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P*D, self.out_ch])  # (P*D)x4 -> 1x(P*D)x4
		bg_raws = torch.cat([bg_raws[...,:-1], self.density_act(bg_raws[..., -1:])], dim=-1)

		tmp = self.f_before(input_fg)
		tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # Mx64

		latent_fg = self.f_after_latent(tmp)  # Mx64
		fg_raw_rgb = self.f_color(latent_fg) # Mx3
		# put back the removed query points, for indices between idx[i] and idx[i+1], put fg_raw_rgb[i] at idx[i]
		fg_raw_rgb_full = torch.zeros((K-1)*P*D, 3, device=fg_raw_rgb.device, dtype=fg_raw_rgb.dtype) # ((K-1)xP*D)x3
		fg_raw_rgb_full[idx] = fg_raw_rgb
		fg_raw_rgb = fg_raw_rgb_full.view([K-1, P*D, 3])  # ((K-1)xP*D)x3 -> (K-1)x(P*D)x3

		fg_raw_shape = self.f_after_shape(tmp) # Mx1
		# if add_blob: # add a gaussian pdf 5 * exp(-||mean||^2 / 2), (with no gradient)
		# 	with torch.no_grad():
		# 		blob = 5 * torch.exp(-torch.norm(fg_means, dim=-1, keepdim=True) ** 2 / 2)
		# 	fg_raw_shape = fg_raw_shape + blob.detach()
		# fill with -inf
		# fg_raw_shape_full = torch.nan_to_num(torch.full((K-1)*P*D, float('-inf'), device=fg_raw_shape.device, dtype=fg_raw_shape.dtype))
		fg_raw_shape_full = torch.zeros((K-1)*P*D, 1, device=fg_raw_shape.device, dtype=fg_raw_shape.dtype) # ((K-1)xP*D)x1
		# fg_raw_shape_full[idx] = fg_raw_shape
		fg_raw_shape_full[idx] = self.density_act(fg_raw_shape)
		fg_raw_shape = fg_raw_shape_full.view([K - 1, P*D])  # ((K-1)xP*D)x1 -> (K-1)x(P*D), density

		fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)x(P*D)x4

		all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # Kx(P*D)x4
		raw_masks = all_raws[..., -1:]
		# raw_masks = self.density_act(all_raws[:, :, -1:])  # Kx(P*D)x1
		# raw_masks = F.relu(all_raws[:, :, -1:], True)  # Kx(P*D)x1
		masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # Kx(P*D)x1

		# print("ratio of fg density above 0.01", torch.sum(masks[1:] > 0.01) / idx.shape[0])
		# print("ratio of bg density above 0.01", torch.sum(masks[:1] > 0.01) / raw_masks[:1].numel())

		raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
		raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

		unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # Kx(P*D)x4
		masked_raws = unmasked_raws * masks
		raws = masked_raws.sum(dim=0)

		return raws, masked_raws, unmasked_raws, masks

	def _build_activation(self, options):
		if options == 'softplus':
			return nn.Softplus
		elif options == 'relu':
			return nn.ReLU
		elif options == 'silu':
			return nn.SiLU
		else:
			assert False, 'activation should be softplus or relu'

