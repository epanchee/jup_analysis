import math
import numpy as np
import matplotlib.animation as animation

class LVCcase:

	tips_loaded = False
	udat_loaded = False
	frames = None
	ani_config = None

	def __init__(self, **kwargs):
		self.project_folder = kwargs['project_folder']
		self.param_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='param.dat')
		self.input_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='input.txt')
		self.data_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='u.dat')
		self.tips_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='tipsHans.info')

	def read_input(self, **kwargs):
		with open(self.input_file, 'r') as f:
		    array = f.read().replace("\n","=").split("=")
		    conf = {array[i] : array[i+1] for i in range(0, len(array) - 1, 2)}
		    self.dT = float(conf['dt']) * int(conf['print_every'])
		    if 'drx' in conf:
		        self.drx = float(conf['drx'])
		        self.dry = float(conf['dry'])
		        self.nx = int(conf['nx']) + 1
		        self.ny = int(conf['ny']) + 1
		        self.nxs = math.floor(self.nx/int(conf['space_mult_x']))
		        self.nys = math.floor(self.ny/int(conf['space_mult_y']))
		        self.dt = float(conf['dt'])
		        self.mult_x = int(conf['space_mult_x'])
		        self.mult_y = int(conf['space_mult_y'])
		    else:
		        self.drx = float(conf['dr'])
		        self.dry = self.drx
		        try:
		            self.nx = int(conf['n']) + 1
		        except:
		            self.params = open(self.param_file, 'r')
		            array = self.params.read().split("\n")
		            self.nx = int(array[2])+1
		        self.ny = self.nx
		        self.nxs = math.floor((self.nx+1)/int(conf['space_mult']))
		        self.nys = self.nxs
		        self.dt = float(conf['dt'])
		        self.mult_x = int(conf['space_mult'])
		        self.mult_y = self.mult_x
		    if 'hole_size_x' in conf:
		    	self.hole_x = int(conf['hole_x'])
		    	self.hole_y = int(conf['hole_y'])
		    	self.hole_size_x = int(conf['hole_size_x'])
		    	self.hole_size_y = int(conf['hole_size_y'])

		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			print("dT = %.2f" % self.dT)
			print("drx = %.2f" % self.drx)
			print("dry = %.2f" % self.dry)
			print("nx = %d" % self.nx)
			print("ny = %d" % self.ny)
			print("nxs = %d" % self.nxs)
			print("nys = %d" % self.nys)
			print("dt = %.3f" % self.dt)
			print("mult_x = %.2f" % self.mult_x)
			print("mult_y = %.2f" % self.mult_y)
			if 'hole_size_x' in conf:
				print("hole_x = %d" % self.hole_x)
				print("hole_y = %d" % self.hole_y)
				print("hole_size_x = %d" % self.hole_size_x)
				print("hole_size_y = %d" % self.hole_size_y)

	def load_tips(self, **kwargs):
		if not self.tips_loaded or 'reload' in kwargs.keys() and kwargs['reload']:
			self.tips = np.loadtxt(self.tips_file)

		self.tips_loaded = True

	def load_udat(self, **kwargs):
		if not self.udat_loaded or 'reload' in kwargs.keys() and kwargs['reload']:
			self.udat = np.loadtxt(self.data_file)
			self.frames_number = int(np.size(self.udat, 0)/self.nxs)
			self.calc_time = float(self.dT * self.frames_number)
		
		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			print("Frames number = %d" % self.frames_number)
			print("Total calculated time = %.2f" % self.calc_time)

		self.udat_loaded = True

	def _get_frame(self, frame_number):
		if not self.udat_loaded:
			raise Exception("udat is not loaded!")
		return self.udat[frame_number * self.nxs : frame_number * self.nxs + self.nxs: 1].transpose()

	def get_frames(self):
		if not self.udat_loaded:
			raise Exception("udat is not loaded!")
		self.frames = list(map(lambda x: self._get_frame(x), range(0, self.frames_number)))

	def get_point_in_time(self, x, y):
		if not self.udat_loaded:
			raise Exception("udat is not loaded!")
		if self.frames == None:
			self.get_frames()
		for i in range(0, self.frames_number):
			yield self.frames[i][y][x]

	def setup_animation(self, **kwargs):
		# bool show_tips, tip_value (-100 by default), bool draw_trace, int K, int first, int last, axes, canvas

		self.ani_config = {
			'show_tips' : False,
			'tip_value' : -100,
			'draw_trace' : False,
			'K' : 1,
			'first' : 0,
			'last' : 100,
			'axes' : None,
			'canvas' : None
		}

		self.path_data = []

		for key in ['show_tips', 'tip_value', 'draw_trace', 'K', 'first', 'last', 'axes']:
			if key in kwargs.keys():
				self.ani_config[key] = kwargs[key]

		if self.frames == None:
			self.get_frames()

		self.frame = self.frames[0]

	def set_canvas(self, _canvas):
		self.canvas = _canvas

	def animate(self, i):
		from matplotlib.patches import Polygon

		self.frame = np.copy(self.frames[i])
		if self.ani_config['show_tips']:
			es = self.tips[self.tips[:,0] == i * self.dT]
			for e in es:
				self.frame[int(e[3] / (self.dry * self.mult_y))][int(e[2] / (self.drx * self.mult_x))] = self.ani_config['tip_value']
				self.path_data.append([e[2], e[3]])

			if self.path_data != [] and self.ani_config['draw_trace']:
				polygon = Polygon(self.path_data, closed=False, fill=False)
				ax.add_patch(polygon)
		self.canvas.set_array(self.frame)
		self.ani_config['axes'].set_title("Time = %d ms" % (self.dT*i))
		return self.canvas,

	def init(self):
		clean_frame = np.full((self.nx,self.ny), np.float64(-84))
		self.canvas.set_array(clean_frame)
		self.ani_config['axes'].set_title("Time = %d ms" % 0)
		return self.canvas,

	def generate_animation(self, figure):
		self.animation = animation.FuncAnimation(figure, self.animate, frames=range(int(self.ani_config['first']/self.dT),int(self.ani_config['last']/self.dT)+1,self.ani_config['K']),
			init_func=self.init, blit=True, repeat=False)

	def save_animation(self, filename):

		# Set up formatting for the movie files
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
		self.animation.save(filename, writer=writer)