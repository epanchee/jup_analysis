import math
import numpy as np
import matplotlib.animation as animation
from matplotlib.pyplot import figure
from numpy import ones,vstack
from numpy.linalg import lstsq
import os, re
from matplotlib.patches import Polygon, Circle, Rectangle

class LVCcase:

	tips_loaded = False
	udat_loaded = False
	stimtable_loaded = False
	stimfield_loaded = False
	frames = None
	ani_config = None
	hole_x_2 = None
	slurm_file = None
	stimtable = None
	plt = None
	conf = {}
	params = {}
	period = None

	def __init__(self, **kwargs):
		self.project_folder = kwargs['project_folder']
		self.param_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='param.dat')
		self.input_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='input.txt')
		self.data_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='u.dat')
		self.tips_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='tipsHans.info')
		self.stimtable_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='stim_table.info')
		self.stimfield_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='stimulus.info')
		self.scar_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='scar.dat')
		self.plt = kwargs['plot']
		self.auto_load = kwargs['auto_load'] if 'auto_load' in kwargs.keys() else False # True | False : if True - automatically loads udat and tips.info if necessary

	def read_input(self, **kwargs):
		with open(self.input_file, 'r') as f:
			s = f.read()
			for a,b in np.array(re.findall(r'(\w+)=([-\w.]+)', s)):
				if 'verbose' in kwargs.keys() and kwargs['verbose']:
					print("params[%s]=%r" % (a,b))
				self.params[a]=b

			array = s.replace("\n","=").split("=")
			self.conf = {array[i] : array[i+1] for i in range(0, len(array) - 1, 2)}
			conf = self.conf
			self.dT = float(conf['dt']) * int(conf['print_every'])
			if 'pace_period' in conf:
				self.period = float(conf['pace_period'])
			if 'drx' in conf:
				self.drx = float(conf['drx'])
				self.dry = float(conf['dry'])
				self.nx = int(conf['nx'])
				self.ny = int(conf['ny'])
				self.nxs = 1+math.floor(self.nx/int(conf['space_mult_x'])) 
				self.nys = 1+math.floor(self.ny/int(conf['space_mult_y']))
				self.dt = float(conf['dt'])
				self.mult_x = int(conf['space_mult_x'])
				self.mult_y = int(conf['space_mult_y'])
			else:
				self.drx = float(conf['dr'])
				self.dry = self.drx
				try:
					self.nx = int(conf['n'])
				except:
					self.params = open(self.param_file, 'r')
					array = self.params.read().split("\n")
					self.nx = int(array[2])
				self.ny = self.nx
				self.nxs = 1+math.floor(self.nx/int(conf['space_mult']))
				self.nys = 1+math.floor(self.ny/int(conf['space_mult']))
				self.dt = float(conf['dt'])
				self.mult_x = int(conf['space_mult'])
				self.mult_y = self.mult_x
			if 'Ttarget' in conf:
				self.Ttarget = float(conf['Ttarget'])
			if 'hole_size_x' in conf:
				self.hole_x = int(conf['hole_x'])
				self.hole_y = int(conf['hole_y'])
				self.hole_size_x = int(conf['hole_size_x'])
				self.hole_size_y = int(conf['hole_size_y'])
			if 'hole_size_x_2' in conf:
				self.hole_x_2 = int(conf['hole_x_2'])
				self.hole_y_2 = int(conf['hole_y_2'])
				self.hole_size_x_2 = int(conf['hole_size_x_2'])
				self.hole_size_y_2 = int(conf['hole_size_y_2'])

		self._job_completed() # check that job has been completed

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
			if 'hole_size_x_2' in conf:
				print("hole_x_2 = %d" % self.hole_x_2)
				print("hole_y_2 = %d" % self.hole_y_2)
				print("hole_size_x_2 = %d" % self.hole_size_x_2)
				print("hole_size_y_2 = %d" % self.hole_size_y_2)
			if 'Ttarget' in conf:
				print("Ttarget = %d" % self.Ttarget)
			if not self.COMPLETED:
				print("\x1b[31m\nTHE JOB WAS NOT COMPLETED!\x1b[0m")


	def _job_completed(self):
		files = os.listdir(self.project_folder)
		
		self.COMPLETED = False
		for file in files:
			if re.match('slurm', file) is not None:
				self.slurm_file = "{experiment_folder}/{filename}".format(experiment_folder=self.project_folder, filename=file)
				if 'Time of calculation' in open(self.slurm_file, 'r').read():
					self.COMPLETED = True


	def load_tips(self, **kwargs):
		if not self.tips_loaded or 'reload' in kwargs.keys() and kwargs['reload']:
			try:
				print("Loading tipsHans.info .....")
				self.tips = np.loadtxt(self.tips_file)
				print("tipsHans.info has been loaded")
			except Exception as e:
				self.tips_loaded = False
				raise Exception("Tips file was not found or it is empty!")
			else:
				self.tips_loaded = True

	def load_udat(self, **kwargs):
		if not self.udat_loaded or 'reload' in kwargs.keys() and kwargs['reload']:
			print("Loading u.dat .....")
			self.udat = np.loadtxt(self.data_file)
			print("u.dat has been loaded")
			self.frames_number = int(np.size(self.udat, 0)/self.nxs)
			self.calc_time = float(self.dT * self.frames_number)
		
		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			print("Frames number = %d" % self.frames_number)
			print("Total calculated time = %.2f" % self.calc_time)

		self.udat_loaded = True

	def load_stimtable(self, reload=False):
		if not self.stimtable_loaded or reload:
			s = open(self.slurm_file, 'r').read()
			stimtable = np.array(re.findall(r'period=(\d+).0 moment_stim=(\d+).0', s))
			self.stimtable = stimtable.astype(np.int)
            
	def load_scar(self):
		self.scar_mask = np.loadtxt(self.scar_file, skiprows=1)
		self.scar_mask = np.transpose(self.scar_mask)

	def draw_tipXY(self, scale=1000, show_stim=False, legend=True):
		if not self.tips_loaded:
			if self.auto_load:
				self.load_tips()
			else:
				raise Exception("tips are not loaded!")

		self.plt.scatter(self.tips[:,0], self.tips[:,2], color='red', s=1, label='x-coordinate (red)')
		self.plt.scatter(self.tips[:,0], self.tips[:,3], color='blue', s=1, label='y-coordinate (blue)')
		self.plt.title('Spiral tip trajectories. X and Y components are shown.')

		if show_stim:
			self.load_stimtable()
			self.plt.gca().vlines(x=self.stimtable[0][1], ymin=0, ymax=100, linewidth=2, color='black', label='first stimulus')
			if hasattr(self, 'Ttarget'):
				firstTtarget = self.stimtable[self.stimtable[:,0] == self.Ttarget][0][1]
				self.plt.gca().vlines(x=firstTtarget, ymin=0, ymax=100, linewidth=2, linestyle='--', color='g', label='first Ttarget stimulus')

		self.plt.ylim(0, max(self.nx * self.drx, self.ny * self.dry))
		if legend:
			self.plt.legend()
		self.plt.grid(True)

	def _get_stim_frame(self, stim_number):
		return self.stimfield[stim_number * self.nx : stim_number * self.nx + self.nx: 1].transpose()

	def _get_tips(self):
		if not self.tips_loaded:
			self.load_tips()
		return self.tips

	def load_stim(self, **kwargs):
		if not self.stimfield_loaded or 'reload' in kwargs.keys() and kwargs['reload']:
			self.stimtable = np.loadtxt(self.stimtable_file, ndmin=2)
			self.stimfield = np.loadtxt(self.stimfield_file)

			self.stim_frames = list(map(lambda x: self._get_stim_frame(x), range(0, self.stimtable[:,2].size)))

	def _get_frame(self, frame_number):
		if not self.udat_loaded:
			if self.auto_load:
				self.load_udat(verbose=True)
			else:
				raise Exception("udat is not loaded!")
		return self.udat[frame_number * self.nxs : frame_number * self.nxs + self.nxs: 1].transpose()

	def get_frames(self):
		if not self.udat_loaded:
			if self.auto_load:
				self.load_udat(verbose=True)
			else:
				raise Exception("udat is not loaded!")
		self.frames = list(map(lambda x: self._get_frame(x), range(0, self.frames_number)))

	def get_point_in_time(self, x, y):
		if self.frames == None:
			self.get_frames()
		for i in range(0, self.frames_number):
			yield self.frames[i][y][x]

	def get_line_solution(self, point1, point2):
		points = [point1, point2]
		x_coords, y_coords = zip(*points)
		A = vstack([x_coords,ones(len(x_coords))]).T
		m, c = lstsq(A, y_coords, rcond=None)[0]
	#     print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
		return (m,c)

	def solve_lnsq(self, line, level):
		# Solving following system of linear equation
		# mx - 1y = -c
		# 0x + 1y = level
		m, c = line
		a = np.array([[m, -1],[0,1]])
		b = np.array([-c, level])
		return np.linalg.solve(a,b)

	def get_cross_points(self, func, level, scale, shift, display_flag):
		points = []
		asc = None # ascending?
		for i in range(len(func) - 1):	                
			if func[i] < level and func[i+1] >= level or func[i] > level and func[i+1] <= level:
				if asc == None:
					if func[i] < func[i+1]:
						asc = True
					else:
						asc = False
				left = (shift+i)*scale
				right = (shift+i+1)*scale	            
				m,c =self.get_line_solution((left, func[i]), (right, func[i+1]))
				x = np.linspace(left, right)
				cross = self.solve_lnsq([m,c], level)
				points.append(cross)
				if display_flag:
					self.plt.plot(x, m*x+c, color='black')
					self.plt.plot(cross[0], cross[1], marker='o', color='black')
			
		return asc, points

	def preprocess_calculation(self, **kwargs):
		# x, y, [start], u_level, scale, [show_plot], [plot]

		start_t = kwargs['start'] if 'start' in kwargs.keys() else 0
		end_t = kwargs['end'] if 'end' in kwargs.keys() else self.frames_number
		u_level = kwargs['u_level']
		scale = kwargs['scale']
		data = kwargs['data']
		display_flag = kwargs['show_plot'] if 'show_plot' in kwargs.keys() else False
		# self.plt = kwargs['plot'] if display_flag else None

		if display_flag:
			figure(num=None, figsize=(12, 4), dpi=80, facecolor='w')
			self.plt.plot([start_t*scale, end_t*scale],[u_level, u_level], color='red')
			self.plt.plot((np.arange(len(data)) + start_t)*scale, data)
			self.plt.grid()

		return self.get_cross_points(data, u_level, scale, start_t, display_flag)

	def calculate_periods_by_tips(self, **kwargs):
		kwargs['scale'] = self.dT
		tips = self._get_tips()
		tips = tips[(tips[:,0] > kwargs['start']) & (tips[:,0] < kwargs['end'])]
		kwargs['data'] = tips[:, 2 if kwargs['coordinate'] == 'x' else 3]
		if 'filter_high' in kwargs.keys():
			kwargs['data'] = kwargs['data'][kwargs['data'] < kwargs['filter_high']]
		if 'filter_low' in kwargs.keys():
			kwargs['data'] = kwargs['data'][kwargs['data'] > kwargs['filter_low']]
		asc, pts = self.preprocess_calculation(**kwargs)

		periods = []
		times = []
		for i in range(0, len(pts)-3, 2):
			periods.append(pts[i+2][0] - pts[i][0])
			times.append(pts[i][0])

		return times, periods

	def calculate_periods(self, **kwargs):
		kwargs['scale'] = self.dT
		if not 'end' in kwargs.keys():
			kwargs['data'] = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']:]
		else:
			kwargs['data'] = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']:kwargs['end']]
		asc, pts = self.preprocess_calculation(**kwargs)

		periods = []
		times = []
		for i in range(0, len(pts)-3, 2):
			periods.append(pts[i+2][0] - pts[i][0])
			times.append(pts[i][0])

		return times, periods

	# def calculate_APD_X(self, **kwargs):
	# 	kwargs['scale'] = self.dT
	# 	kwargs['data'] = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']:]
	# 	asc, pts = self.preprocess_calculation(**kwargs)

	# 	start = 0 if asc else 1

	# 	apds = []
	# 	times = []
	# 	for i in range(start, len(pts)-2, 2):
	# 		apds.append(pts[i+1][0] - pts[i][0])
	# 		times.append(pts[i][0])

	# 	return times, apds

	def calculate_APD_X(self, **kwargs):
		times, apds, dis = self.calculate_APD_X_with_DI(**kwargs)
		return times, apds

	def calculate_APD_by_percentage(self, **kwargs):
		times, apds, dis = self.calculate_APD_by_percentage_with_DI(**kwargs)
		return times, apds

	def calculate_APD_X_with_DI(self, **kwargs):
		kwargs['scale'] = self.dT
		if 'end' in kwargs.keys():
			_data = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']:kwargs['end']]
		else:
			_data = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']: int(self.calc_time)]
		kwargs['data'] = _data
		asc, pts = self.preprocess_calculation(**kwargs)

		start = 0 if asc else 1

		apds = []
		dis = []
		times = []
		for i in range(start, len(pts)-1, 2):
			dis.append(pts[i][0] - pts[i-1][0])
			apds.append(pts[i+1][0] - pts[i][0])
			times.append(pts[i][0])

		return times, apds, dis

	def calculate_APD_by_percentage_with_DI(self, **kwargs):
		APD_percent = kwargs['APD_percent']

		if 'end' in kwargs.keys():
			_data = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']:kwargs['end']]
		else:
			_data = list(self.get_point_in_time(kwargs['x'], kwargs['y']))[kwargs['start']: int(self.calc_time)]
		peak = np.max(_data)
		min_value = np.min(_data)
		APD_level = peak - (peak - min_value)*APD_percent
		print("Max is %.2f, min is %.2f, APD_level %.2f" % (peak, min_value, APD_level))

		kwargs['u_level'] = APD_level
		return self.calculate_APD_X_with_DI(**kwargs)

	def get_Yline(self, FRAME, y):
		if not self.udat_loaded:
			if self.auto_load:
				self.load_udat(verbose=True)
			else:
				raise Exception("udat is not loaded!")
		return self.frames[FRAME][y]

	def calculate_wavelength(self, **kwargs):
		kwargs['scale'] = self.drx*self.mult_x
		kwargs['data'] = self.get_Yline(kwargs['frame_number'], kwargs['y'])
		asc, pts = self.preprocess_calculation(**kwargs)

		start = 1 if asc else 0

		wls = []
		xs = []
		for i in range(start, len(pts)-2, 2):
			wls.append(pts[i+1][0] - pts[i][0])
			xs.append(pts[i][0])

		return xs, wls

	def setup_animation(self, **kwargs):
		# bool show_tips, bool draw_trace, int K, int first, int last, plot, double vmax, bool ani_title

		self.ani_config = {
			'show_tips' : False,
			'draw_trace' : False,
			'obstacle' : False,
			'K' : 1,
			'first' : 0,
			'last' : 100,
			'canvas' : None,
			'plot' : None,
			'vmax' : 25,
			'ani_title' : True,
			'axis_auto_scale' : True
		}

		self.path_data = []

		for key in self.ani_config.keys():
			if key in kwargs.keys():
				self.ani_config[key] = kwargs[key]

		# self.plt = self.ani_config['plot']
		self.ax = self.plt.gca()

		if self.frames == None:
			self.get_frames()

		if self.ani_config['last']/self.dT >= self.frames_number:
			print("There are %d frames availible. Variable 'last' has been set to %d ms" % (self.frames_number, (self.frames_number - 1) * self.dT))
			self.ani_config['last'] = (self.frames_number - 1) * self.dT

		self.frame = self.frames[0]
		self.patches_tips = []

		if self.ani_config['canvas'] != None:
			self.canvas = self.ani_config['canvas']
		else:
			self.canvas = self.plt.imshow(self.frame, origin='lower', interpolation='none', extent=
				[0, 1.01 * self.nx * self.drx, 0, 1.01 * self.ny * self.dry], animated=True, vmin=np.amin(self.frames), vmax=self.ani_config['vmax'])
		
		if not self.ani_config['axis_auto_scale']:
			self.ax.set_xticks(np.arange(0, 1.01 * self.ny * self.dry, 20))
			self.ax.set_yticks(np.arange(0, 1.01 * self.ny * self.dry, 20))

		if self.ani_config['obstacle']:
			pass

	def draw_frame(self, **kwargs):
		from matplotlib import cm
		
		self.ax = self.plt.gca()
		vmax = kwargs['vmax'] if 'vmax' in kwargs.keys() else 25
		obstacle_color = kwargs['obstacle_color'] if 'obstacle_color' in kwargs.keys() else 'white'
		cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else cm.get_cmap('viridis')

		if self.frames == None:
			self.get_frames()

		frame = np.copy(self.frames[kwargs['N']])
		tip_radius = kwargs['tip_radius'] if 'tip_radius' in kwargs.keys() else 0.3
		
		if 'show_tips' in kwargs.keys() and kwargs['show_tips']:
			self.load_tips()
			if self.tips.size >0:
				patches_tips = []
				es = self.tips[self.tips[:,0] == kwargs['N'] * self.dT]
				for e in es:
					tip_patch = Circle(xy=(e[2], e[3]), radius=tip_radius, fill=True, color='black')
					patches_tips.append(tip_patch)
					self.ax.add_patch(tip_patch)

		if 'axis_auto_scale' in kwargs.keys() and not kwargs['axis_auto_scale']:
			self.ax.set_xticks(np.arange(0, 1.01 * self.nx * self.drx, 20))
			self.ax.set_yticks(np.arange(0, 1.01 * self.ny * self.dry, 20))

		if 'set_title' in kwargs.keys() and kwargs['set_title']:
			time_shift = 0
			if 'time_shift' in kwargs.keys():
				time_shift = kwargs['time_shift']
			self.ax.set_title("Time = %d ms" % (time_shift + self.dT*kwargs['N']))
            
		if 'obstacle' in kwargs.keys() and kwargs['obstacle']:
			self.load_scar()
			mask = self.scar_mask
			stencil_x, stencil_y = int(self.params['scar_stencil_j']), int(self.params['scar_stencil_i'])
			real_mask = [[0]*self.ny for i in range(self.nx)]
			cmap.set_under(obstacle_color)
			
			for i in range(self.nx):
				for j in range(self.ny):
					real_mask[i][j] = 0
					
			for i in range(len(mask)):
				for j in range(len(mask[i])):
					real_mask[stencil_x + i][stencil_y + j] = mask[i][j]
			
			for i in range(len(frame)):
				for j in range(len(frame[i])):
					if(real_mask[i*self.mult_x][j*self.mult_y] == 1):
						frame[i][j] = -100
                                  
		self.plt.imshow(frame, origin='lower', interpolation='none', extent=[0, 1.01 * self.nx * self.drx, 0, 1.01 * self.ny * self.dry], 
			vmin=np.amin(self.frames), cmap=cmap, vmax=vmax)

	def animate(self, i):
		self.frame = np.copy(self.frames[i])

		if not self.tips_loaded and self.auto_load:
				self.load_tips()
		
		if self.tips_loaded and self.tips.size > 0 and self.ani_config['show_tips']:
			
			# clear tips
			for tip in self.patches_tips:
				tip.set_visible(False)

			self.patches_tips = []
			es = self.tips[self.tips[:,0] == i * self.dT]
			for e in es:
				tip_patch = Circle(xy=(e[2], e[3]), radius=0.3, fill=True, color='black')
				self.patches_tips.append(tip_patch)
				self.ax.add_patch(tip_patch)
				self.path_data.append([e[2], e[3]])

			if self.path_data != [] and self.ani_config['draw_trace']:
				polygon = Polygon(self.path_data, closed=False, fill=False)
				self.ax.add_patch(polygon)

		self.canvas.set_array(self.frame)
		self.ax.set_title("Time = %d ms" % (self.dT*i)) if self.ani_config['ani_title'] else None
		return self.canvas,

	def init(self):
		clean_frame = np.full((self.nx,self.ny), np.float64(-84))
		self.canvas.set_array(clean_frame)
		self.ax.set_title("Time = %d ms" % 0) if self.ani_config['ani_title'] else None
		return self.canvas,

	def generate_animation(self, fig):
		self.animation = animation.FuncAnimation(fig, self.animate, frames=range(int(self.ani_config['first']/self.dT),int(self.ani_config['last']/self.dT)+1,self.ani_config['K']),
			init_func=self.init, blit=True, repeat=False)

	def save_animation(self, **kwargs):
		# filename, overwrite, generate_filename, folder

		if 'generate_filename' in kwargs.keys() and kwargs['generate_filename']:
			proj_id, proj_prefix = self.project_folder.split('/')[-2:]
			if self.period == None:
				kwargs['filename'] = "{project_id}_{project_prefix}_{first}-{last}.mp4".format(
					project_id=proj_id, project_prefix=proj_prefix, first=self.ani_config['first'], last=self.ani_config['last']
				)
			else:
				kwargs['filename'] = "{project_id}_{project_prefix}_{period}_{first}-{last}.mp4".format(
					project_id=proj_id, project_prefix=proj_prefix, period=self.period, first=self.ani_config['first'], last=self.ani_config['last']
				)

		if 'folder' in kwargs.keys() and not kwargs['folder'] == '':
			kwargs['filename'] = "%s/%s" % (kwargs['folder'], kwargs['filename'])

		if os.path.isfile(kwargs['filename']):
			if not 'overwrite' in kwargs.keys() or not kwargs['overwrite']:
				print("File %s exists! To overwrite this file set option 'overwrite=True'." % kwargs['filename'])
				return

		if not 'fps' in kwargs.keys():
			kwargs['fps'] = 30

		# Set up formatting for the movie files
		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=kwargs['fps'], metadata=dict(artist='Me'), bitrate=1800)
		self.animation.save(kwargs['filename'], writer=writer)

		print("Animation has been saved to file %s" % kwargs['filename'])
