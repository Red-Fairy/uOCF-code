"""This module contains simple helper functions """
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import itertools

import six
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import skimage.transform
import random
import colorsys
import torch.nn.functional as F
from matplotlib import patches,  lines

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

def write_location(f, location, i, description=''):
	f.write(f"Test image {i}: {description}\n")
	location = location.cpu().numpy()
	for i in range(location.shape[0]):
		f.write(f"Slot {i}:, ({location[i][0]:.4f}, {location[i][1]:.4f})\n")
	f.flush()

def resize_masks(masks, image_size):
	"""
	Resize masks size
	:param masks: tensor of shape (n, 1, h, w)
	:param image_size: H, W
	:return: numpy array of shape (n, H, W)
	"""
	masks_n = masks.squeeze()
	masks_resize = np.zeros((masks_n.shape[0], image_size[0], image_size[1]))

	for i in range(masks_n.shape[0]):
		masks_resize[i] = skimage.transform.resize(masks_n[i], image_size, order=3)
		masks_resize[i] = (masks_resize[i]>=0.75).astype('uint8')
	return masks_resize

def mask2bbox(mask):
	inds = (np.where(mask==1))
	xmin, xmax = inds[1].min(), inds[1].max()
	ymin, ymax = inds[0].min(), inds[0].max()
	width = xmax - xmin
	height = ymax - ymin
	rect = patches.Rectangle((xmin, ymin),xmax - xmin, ymax - ymin,linewidth=1,edgecolor='r',facecolor='none')
	return (xmin + width / 2, ymin + height / 2, width, height), rect


def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1,
								  image[:, :, c] *
								  (1 - alpha) + \
								  alpha * color[c] * 255,
								  image[:, :, c])
	return image

def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors


def display_image(image, masks, display_index=False):
	image_mask = image
	colors = random_colors(masks.shape[0])
	for i in range(masks.shape[0]):
		image_mask = apply_mask(image, masks[i], colors[i])
	if display_index:
		image_mask = Image.fromarray(image_mask)
		draw = ImageDraw.Draw(image_mask)
		for i in range(masks.shape[0]):
			min_pixel = 30
			if masks[i].sum() > min_pixel:
				mask = masks[i]
				mask_eroded = np.array(Image.fromarray(mask).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)))
				if mask_eroded.sum() <= min_pixel - 20:
					continue
				(x_center, y_center, _, _), rect = mask2bbox(mask_eroded)
				# draw.text((x_center, y_center), str(i), (0, 0, 0))
				# import ipdb; ipdb.set_trace()
				draw.rectangle(list(rect.get_bbox().get_points().reshape(-1)))
		image_mask = np.array(image_mask)


	return image_mask

def get_colormap(rgb=True):
	color_list = np.array(
		[
			0, 0, 0.5625
			, 0, 0, 0.6250
			, 0, 0, 0.6875
			, 0, 0, 0.7500
			, 0, 0, 0.8125
			, 0, 0, 0.8750
			, 0, 0, 0.9375
			, 0, 0, 1.0000
			, 0, 0.0625, 1.0000
			, 0, 0.1250, 1.0000
			, 0, 0.1875, 1.0000
			, 0, 0.2500, 1.0000
			, 0, 0.3125, 1.0000
			, 0, 0.3750, 1.0000
			, 0, 0.4375, 1.0000
			, 0, 0.5000, 1.0000
			, 0, 0.5625, 1.0000
			, 0, 0.6250, 1.0000
			, 0, 0.6875, 1.0000
			, 0, 0.7500, 1.0000
			, 0, 0.8125, 1.0000
			, 0, 0.8750, 1.0000
			, 0, 0.9375, 1.0000
			, 0, 1.0000, 1.0000
			, 0.0625, 1.0000, 0.9375
			, 0.1250, 1.0000, 0.8750
			, 0.1875, 1.0000, 0.8125
			, 0.2500, 1.0000, 0.7500
			, 0.3125, 1.0000, 0.6875
			, 0.3750, 1.0000, 0.6250
			, 0.4375, 1.0000, 0.5625
			, 0.5000, 1.0000, 0.5000
			, 0.5625, 1.0000, 0.4375
			, 0.6250, 1.0000, 0.3750
			, 0.6875, 1.0000, 0.3125
			, 0.7500, 1.0000, 0.2500
			, 0.8125, 1.0000, 0.1875
			, 0.8750, 1.0000, 0.1250
			, 0.9375, 1.0000, 0.0625
			, 1.0000, 1.0000, 0
			, 1.0000, 0.9375, 0
			, 1.0000, 0.8750, 0
			, 1.0000, 0.8125, 0
			, 1.0000, 0.7500, 0
			, 1.0000, 0.6875, 0
			, 1.0000, 0.6250, 0
			, 1.0000, 0.5625, 0
			, 1.0000, 0.5000, 0
			, 1.0000, 0.4375, 0
			, 1.0000, 0.3750, 0
			, 1.0000, 0.3125, 0
			, 1.0000, 0.2500, 0
			, 1.0000, 0.1875, 0
			, 1.0000, 0.1250, 0
			, 1.0000, 0.0625, 0
			, 1.0000, 0, 0
			, 0.9375, 0, 0
			, 0.8750, 0, 0
			, 0.8125, 0, 0
			, 0.7500, 0, 0
			, 0.6875, 0, 0
			, 0.6250, 0, 0
			, 0.5625, 0, 0
			, 0.5000, 0, 0
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3))
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list

def get_cyan_to_pink_colormap(steps=64, rgb=True):
	colors = []
	
	# Define the starting (cyan) and ending (pink) colors
	cyan = np.array([0, 255, 255], dtype=np.float32)
	pink = np.array([255, 105, 180], dtype=np.float32)
	
	for i in range(steps):
		t = i / (steps - 1)  # Normalized step in [0, 1]
		color = (1 - t) * cyan + t * pink
		colors.append(color)
	
	colormap = np.array(colors, dtype=np.float32)
	
	if not rgb:
		colormap = colormap[:, ::-1]
	
	return colormap

def tensor2im(input_image, imtype=np.uint8, use_color_map=True):
	""""Converts a Tensor array into a numpy image array.

	Parameters:
		input_image (tensor) --  the input image tensor array, range=[-1,1], CxHxW
		imtype (type)        --  the desired type of the converted numpy array
		use_color_map: if True, when inputting grayscale (n_ch==1), do color mapping
	output:
		image_numpy: HxWx3
	"""
	if not isinstance(input_image, np.ndarray):
		if isinstance(input_image, torch.Tensor):  # get the data from a variable
			image_tensor = input_image.data
		else:
			return input_image
		image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
		image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0  # now HxWx3, [0,1]
		if image_numpy.shape[2] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (1, 1, 3))
			if use_color_map:
				mapped = image_numpy.copy()
				colormap = get_colormap()  # 64-bin color map, 64x3
				img = image_numpy[:, :, 0]
				grid = np.linspace(0, 1, 65)
				for i in range(64):
					lower, upper = grid[i], grid[i + 1]
					img_pos = (img <= upper) & (img >= lower)
					mapped[img_pos, :] = colormap[i, :]
				image_numpy = mapped
		image_numpy *= 255.0
		image_numpy = image_numpy.astype(imtype)
	else:  # if it is a numpy array, do nothing
		image_numpy = input_image.astype(imtype)
	return image_numpy

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	torch.backends.cuda.matmul.allow_tf32 = False
	torch.backends.cudnn.allow_tf32 = False
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

def diagnose_network(net, name='network'):
	"""Calculate and print the mean of average absolute(gradients)

	Parameters:
		net (torch network) -- Torch network
		name (str) -- the name of the network
	"""
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path):
	"""Save a numpy image to the disk

	Parameters:
		image_numpy (numpy array) -- input numpy array
		image_path (str)          -- the path of the image
	"""
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
	"""Print the mean, min, max, median, std, and size of a numpy array

	Parameters:
		val (bool) -- if print the values of the numpy array
		shp (bool) -- if print the shape of the numpy array
	"""
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def print_tensor(x):
	print('mean={}, min={}, max={}, median={}, std={}'.format(x.mean(), x.min(), x.max(), x.median(), x.std()))

def mkdirs(paths):
	"""create empty directories if they don't exist

	Parameters:
		paths (str list) -- a list of directory paths
	"""
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)


def mkdir(path):
	"""create a single empty directory if it didn't exist

	Parameters:
		path (str) -- a single directory path
	"""
	if not os.path.exists(path):
		os.makedirs(path)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	val = 0
	avg = 0
	sum = 0
	count = 0
	tot_count = 0

	def __init__(self):
		self.reset()
		self.tot_count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.tot_count += n
		self.avg = self.sum / self.count

class GroupMeters(object):
	def __init__(self):
		self._meters = collections.defaultdict(AverageMeter)

	def reset(self):
		list(map((AverageMeter.reset, self._meters.values())))

	def update(self, updates=None, value=None, n=1, **kwargs):
		"""
		Example:
			>>> meters.update(key, value)
			>>> meters.update({key1: value1, key2: value2})
			>>> meters.update(key1=value1, key2=value2)
		"""
		if updates is None:
			updates = {}
		if updates is not None and value is not None:
			updates = {updates: value}
		updates.update(kwargs)
		for k, v in updates.items():
			self._meters[k].update(v, n=n)

	def __getitem__(self, name):
		return self._meters[name]

	def items(self):
		return self._meters.items()

	@property
	def sum(self):
		return {k: m.sum for k, m in self._meters.items() if m.count > 0}

	@property
	def avg(self):
		return {k: m.avg for k, m in self._meters.items() if m.count > 0}

	@property
	def val(self):
		return {k: m.val for k, m in self._meters.items() if m.count > 0}

	def format(self, caption, values, kv_format, glue):
		meters_kv = self._canonize_values(values)
		log_str = [caption]
		log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
		return glue.join(log_str)

	def format_simple(self, caption, values='avg', compressed=True):
		if compressed:
			return self.format(caption, values, '{}={:4f}', ' ')
		else:
			return self.format(caption, values, '\t{} = {:4f}', '\n')

	def _canonize_values(self, values):
		if isinstance(values, six.string_types):
			assert values in ('avg', 'val', 'sum')
			meters_kv = getattr(self, values)
		else:
			meters_kv = values
		return meters_kv

def gen_cam2world(cam_pos, origin=(0,0,0)):
	'''
	cam_pos: 3D position of the camera
	origin: 3D position of the origin
	'''
	x, y, z = cam_pos

	# Calculte the cam2world matrix for the camera positioned at (x, y, z), pointing at (origin[0], origin[1], origin[2])
	forward_direction = torch.tensor([origin[0]-x, origin[1]-y, origin[2]-z], dtype=torch.float32)
	forward_direction = forward_direction / torch.norm(forward_direction)

	# world up vector is (0, 1, 0)
	# up_direction = torch.tensor([x*z/(x**2 + y**2), y*z/(x**2 + y**2), -1], dtype=torch.float32)
	up_direction = torch.tensor([(x-origin[0])*(z-origin[2])/((x-origin[0])**2 + (y-origin[1])**2), (y-origin[1])*(z-origin[2])/((x-origin[0])**2 + (y-origin[1])**2), -1], dtype=torch.float32)
	up_direction = up_direction / torch.norm(up_direction)
	right_direction = torch.cross(up_direction, forward_direction)

	# Construct the cam2world matrix
	rotation_matrix = torch.stack([right_direction, up_direction, forward_direction], dim=1)
	# flip the y axis (we use left hand coordinate system)
	# rotation_matrix[:, 1] = -rotation_matrix[:, 1]
	translation_vector = torch.tensor([x, y, z], dtype=torch.float32).view(3, 1)
	cam2world_matrix = torch.cat([rotation_matrix, translation_vector], dim=1)
	cam2world_matrix = torch.cat([cam2world_matrix, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0) # [4, 4]

	return cam2world_matrix

def get_cam2world_from_path(path, n_views, origin=(0,0,0)):
    cam2world_input = torch.from_numpy(np.loadtxt(path))
    x, y, z = cam2world_input[0, 3], cam2world_input[1, 3], cam2world_input[2, 3]
    radius_xy, azimuth_xy = torch.sqrt(x ** 2 + y ** 2).item(), torch.atan2(y, x).item()

    n_azimuths = n_views
    azimuths = torch.linspace(0, 2*np.pi, n_azimuths+1)[:-1] + azimuth_xy
    # generate camera positions
    cam_positions = torch.stack([radius_xy * torch.cos(azimuths), radius_xy * torch.sin(azimuths), z * torch.ones(n_azimuths)], dim=1)
    # generate cam2world matrices
    cam2world_matrices = torch.stack([gen_cam2world(cam_positions[i]) for i in range(n_azimuths)], dim=0)
    return cam2world_matrices

def get_spiral_cam2world(radius, height, angle_range=(0, 360), n_views=48, radians=True, height_range=(0.8, 1.25), radius_range=(1., 1.), origin=(0, 0, 0)):
	"""
	Get spiral camera to world matrix
	radius: radius of the spiral
	height: height of the spiral
	n_views: number of views
	angle_range: range of the angle
	return: Tensor of shape (n_views, 4, 4)
	"""
	# Convert theta to radians
	if not radians:
		theta = np.radians(theta)
		# angle_range = (np.radians(angle_range[0]), np.radians(angle_range[1]))

	i1, i2, i3, i4 = 3 * n_views // 10, n_views // 2, 7 * n_views // 10, n_views

	# Calculate the rotation angle for each view
	if angle_range[0] != angle_range[1]:
		rotation_angles = np.linspace(angle_range[0], angle_range[1], i2)
		rotation_angles = np.append(rotation_angles, np.linspace(angle_range[1], angle_range[0], i4-i2))
		# rotation_angles = np.linspace(angle_range[0], angle_range[1], i1)
		# rotation_angles = np.append(rotation_angles, np.ones(i2-i1) * angle_range[1])
		# rotation_angles = np.append(rotation_angles, np.linspace(angle_range[1], angle_range[0], i3-i2))
		# rotation_angles = np.append(rotation_angles, np.ones(i4-i3) * angle_range[0])
	else:
		rotation_angles = np.ones(n_views) * angle_range[0]

	# prepare z, from 0.75 to 1.5, then from 1.5 to 0.75, then from 0.75 to 1.5, then from 1.5 to 0.75
	h_min, h_max = height_range
	if angle_range[0] != angle_range[1]:
		zs = (h_min + h_max) / 2 + (h_max - h_min) / 2 * np.sin(np.linspace(0, 2*np.pi, n_views))
		# zs = np.zeros(n_views)
		# zs[:i1] = np.linspace(h_min, h_max, i1)
		# zs[i1:i2] = np.linspace(h_max, h_min, i2-i1)
		# zs[i2:i3] = np.linspace(h_min, h_max, i3-i2)
		# zs[i3:] = np.linspace(h_max, h_min, i4-i3)
		zs = zs * height
	else:
		zs = np.ones(n_views) * height
	# print(rotation_angles)

	# Initialize a list to store the transformation matrices
	cam2world_matrices = []

	def i2radius(i):
		return (radius_range[0] + radius_range[1]) / 2 + (radius_range[1] - radius_range[0]) / 2 * np.sin(i / n_views * 2 * np.pi)

	for i, (angle, z) in enumerate(zip(rotation_angles, zs)):
		# Calculate the camera position on the spiral
		x = radius * np.cos(angle) * i2radius(i)
		y = radius * np.sin(angle) * i2radius(i)
		cam2world_matrices.append(gen_cam2world((x, y, z), origin))

	# Convert the list to a tensor
	cam2world_matrices = torch.stack(cam2world_matrices, dim=0)

	return cam2world_matrices

def get_spherical_cam2world(radius, elevation, n_views=48, radians=True, camera_normalize=False, origin=(0, 0, 0), around_table=False):
	"""
	Get spherical camera to world matrix
	radius: radius of the sphere
	theta: angle between the line from the camera to the origin and the XY plane
	n_views: number of views
	return: Tensor of shape (n_views, 4, 4)
	"""

	# Convert elevation to radians
	if not radians:
		elevation = np.radians(elevation)

	# Calculate the rotation angle for each view
	azimuths = np.linspace(0, 2 * np.pi, n_views, endpoint=False)

	# Initialize a list to store the transformation matrices
	cam2world_matrices = []

	for azimuth in azimuths:
		# Calculate the camera position on the sphere
		# if camera_normalize:
		# 	x = radius * np.sin(theta) * np.cos(angle) + origin[0]
		# 	z = radius * np.sin(theta) * np.sin(angle) + origin[2]
		# 	y = radius * np.cos(theta) + origin[1]
		# else:
		x = radius * np.cos(elevation) * np.cos(azimuth)
		y = radius * np.cos(elevation) * np.sin(azimuth)
		z = radius * np.sin(elevation)

		cam2world_matrices.append(gen_cam2world((x, y, z), origin))

	if around_table:
		canonical_pose = torch.Tensor([[-1, 0, 0, 0],
										[0, 0, -1, radius],
										[0, -1, 0, 0],
										[0, 0, 0, 1]])
		rot = torch.matmul(canonical_pose, cam2world_matrices[0].inverse())
		for i in range(len(cam2world_matrices)):
			cam2world_matrices[i] = torch.matmul(rot, cam2world_matrices[i])

	cam2world_matrices = torch.stack(cam2world_matrices, dim=0)

	return cam2world_matrices

def get_predefined_cam2world(root, scene_id, n_views=60, radius=4):
	file_paths = [f'{scene_id*n_views+i:05d}_sc{scene_id:04d}_az{i:02d}_RT.txt' for i in range(n_views)]
	file_paths = [os.path.join(root, file_path) for file_path in file_paths]
	cam2world_matrices = []
	for file_path in file_paths:
		cam2world_matrices.append(torch.from_numpy(np.loadtxt(file_path)).to(torch.float32))
	canonical = torch.Tensor([[-1, 0, 0, 0],
						  [0, 0, -1, radius],
						  [0, -1, 0, 0],
						  [0, 0, 0, 1]]).to(torch.float32)
	rot = torch.matmul(canonical, cam2world_matrices[0].inverse())
	for i in range(len(cam2world_matrices)):
		cam2world_matrices[i] = torch.matmul(rot, cam2world_matrices[i])
	cam2world_matrices = torch.stack(cam2world_matrices, dim=0)
	return cam2world_matrices

def parse_wanted_indice(wanted_indices):
	if wanted_indices is None:
		return None
	if isinstance(wanted_indices, str):
		if '-' in wanted_indices:
			wanted_indices = wanted_indices.split('-')
			wanted_indices = [int(i) for i in wanted_indices]
			wanted_indices = list(range(wanted_indices[0], wanted_indices[1]+1))
		else:
			wanted_indices = wanted_indices.split(',')
			wanted_indices = [int(i) for i in wanted_indices]
	return wanted_indices





