from .base_options import BaseOptions


class TestOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)  # define shared options
		parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
		parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
		parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
		# Dropout and Batchnorm has different behavioir during training and test.
		parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
		parser.add_argument('--testset_name', type=str, default='testset')
		parser.add_argument('--stage', type=str, default='fine', help='stage of the model')
		parser.add_argument('--n_objects_eval', type=int,default=None, help='number of objects for manipulation')
		parser.add_argument('--video', action='store_true', help='only visualize the results, no quantitative evaluation')
		parser.add_argument('--visual_idx', type=int, default=0, help='index of the image to visualize')
		parser.add_argument('--recon_only', action='store_true', help='only visualize the reconstruction')
		parser.add_argument('--video_mode', type=str, default='spherical', help='spherical or spiral')
		parser.add_argument('--move2center', action='store_true', help='move the object to the center of the image')
		parser.add_argument('--wanted_indices', type=str, default=None, help='indices of the images to visualize')
		parser.add_argument('--show_recon_stats', action='store_true', help='show the statistics of the reconstruction')
		parser.add_argument('--vis_disparity', action='store_true', help='visualize the disparity map')
		parser.add_argument('--vis_render_disparity', action='store_true', help='visualize the disparity map')
		parser.add_argument('--vis_attn', action='store_true', help='visualize the attention map')
		parser.add_argument('--vis_mask', action='store_true', help='visualize the mask')
		parser.add_argument('--vis_render_mask', action='store_true', help='visualize the mask')
		parser.add_argument('--no_loss', action='store_true')
		self.isTrain = False
		return parser