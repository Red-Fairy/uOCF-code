from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	"""This class includes training options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# visdom and HTML visualization parameters
		parser.add_argument('--display_grad', action='store_true')
		parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
		parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
		parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
		# network saving and loading parameters
		parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
		parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
		parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
		parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		# training parameters
		parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
		parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
		parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
		parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate for adam')
		parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
		parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
		# load pretrained networks
		parser.add_argument('--load_pretrain', action='store_true', help='load the pretrained model from the specified location')
		parser.add_argument('--load_epoch', type=str, default='latest', help='load the pretrained model from the specified epoch')
		parser.add_argument('--load_pretrain_path', type=str, default=None, help='load the pretrained model from the specified location')
		
		parser.add_argument('--load_encoder', type=str, default='unload', help='load encoder')
		parser.add_argument('--load_slotattention', type=str, default='unload', help='load slotattention')
		parser.add_argument('--load_decoder', type=str, default='unload', help='load decoder')

		# one object to multiple objects
		parser.add_argument('--one2four', action='store_true', help='one2four')
		parser.add_argument('--stratified', action='store_true', help='stratified sampling')
		parser.add_argument('--large_decoder_lr', action='store_true', help='large decoder lr')
		parser.add_argument('--depth_supervision', action='store_true', help='use depth supervision')
		parser.add_argument('--collapse_prevent', type=int, default=0, help='collapse prevent iterations')
		
		# uOCF - transformer
		parser.add_argument('--vis_mask', action='store_true', help='visualize the mask')

		self.isTrain = True
		return parser
