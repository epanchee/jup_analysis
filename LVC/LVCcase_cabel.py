import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, re
from matplotlib.pyplot import figure
from numpy import ones,vstack
from numpy.linalg import lstsq

class CabelCase:

	udat_loaded = False
	stimtable_loaded = False
	stimtable = None
	conf = {}

	def __init__(self, **kwargs):
		self.project_folder = kwargs['project_folder']
		self.param_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='param.dat')
		self.input_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='input.txt')
		self.data_file = "{experiment_folder}/{filename}".format(experiment_folder=kwargs['project_folder'], filename='u.dat')

	def read_input(self, **kwargs):
		with open(self.input_file, 'r') as f:
			array = f.read().replace("\n","=").split("=")
			conf = {array[i] : array[i+1] for i in range(0, len(array) - 1, 2)}
			self.conf = conf
			self.dT = float(conf['dt']) * int(conf['print_every'])
			self.period = float(conf['pace_period'])
			if 'pace_stop' in conf:
				self.pace_stop = int(conf['pace_stop'])
			if 't_S' in conf:
				self.t_S = int(conf['t_S'])
			self.dr = float(conf['dr'])
			self.N = int(conf['N'])
			self.ns = math.floor(self.N/int(conf['space_mult']))
			self.dt = float(conf['dt'])
			self.mult = int(conf['space_mult'])
			if 'slope' in conf:
				self.slope = float(conf['slope'])

		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			print("dT = %.2f" % self.dT)
			print("dr = %.2f" % self.dr)
			print("N = %d" % self.N)
			print("ns = %d" % self.ns)
			print("dt = %.3f" % self.dt)
			print("space_mult = %.2f" % self.mult)
			if 'pace_stop' in conf:
				print("pace_stop = %d" % self.pace_stop)
			if 't_S' in conf:
				print("t_S = %d" % self.t_S)
			if 'slope' in conf:
				print("slope = %.1f" % self.slope)

			if not self._job_completed():
				print("\x1b[31m\nTHE JOB WAS NOT COMPLETED!\x1b[0m")

	def _job_completed(self):
		files = os.listdir(self.project_folder)
		self.slurm_file = None
		
		for file in files:
			if re.search('slurm', file):
				self.slurm_file = "{experiment_folder}/{filename}".format(experiment_folder=self.project_folder, filename=file)
				break

		if self.slurm_file is not None:
			if 'A wave was' in open(self.slurm_file, 'r').read():
				return True
			else:
				return False

	def load_udat(self, **kwargs):
		if not self.udat_loaded or 'reload' in kwargs.keys() and kwargs['reload']:
			self.udat = np.loadtxt(self.data_file)
			self.frames_number = int(np.size(self.udat, 0))
			self.calc_time = float(self.dT * self.frames_number)
		
		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			print("Total calculated time = %.2f" % self.calc_time)

		self.udat_loaded = True
		self.time = int(self.calc_time)

	def load_stimtable(self, reload=False):
		if not self.stimtable_loaded or reload:
			self._job_completed()
			s = open(self.slurm_file, 'r').read()
			stimtable = np.array(re.findall(r'period=(\d+).0 moment_stim=(\d+).0', s))
			self.stimtable = stimtable.astype(np.int)

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

	def get_cross_points(self, func, level, scale, shift, display_flag, print_flag):
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

	            if print_flag:
	            	print("Point %.2f " % cross[0])
	            if display_flag:
		            self.plt.plot(x, m*x+c, color='black')
		            self.plt.plot(cross[0], cross[1], marker='o', color='black')
	        
	    return asc, points

	def preprocess_calculation(self, **kwargs):
		# x, y, [start], u_level, scale, [plot]

		start_t = kwargs['start'] if 'start' in kwargs.keys() else 0
		u_level = kwargs['u_level']
		scale = kwargs['scale']
		data = kwargs['data']
		display_flag = False
		print_flag = kwargs['print_points'] if 'print_points' in kwargs.keys() else False
		if 'plot' in kwargs.keys():
			self.plt = kwargs['plot']
			display_flag = True 
		else:
			self.plt = None

		if display_flag:
			figure(num=None, figsize=(12, 4), dpi=80, facecolor='w')
			self.plt.plot([start_t*scale, (start_t+len(data))*scale],[u_level, u_level], color='red')
			self.plt.plot((np.arange(len(data))+start_t)*scale, data)
			self.plt.grid()

		return self.get_cross_points(data, u_level, scale, start_t, display_flag, print_flag)

	def calculate_periods(self, **kwargs):
		self.load_udat()
		kwargs['scale'] = self.dT
		kwargs['data'] = list(self.udat[:,kwargs['x']])[kwargs['start']:]
		asc, pts = self.preprocess_calculation(**kwargs)

		periods = []
		times = []
		for i in range(0, len(pts)-2, 2):
			periods.append(pts[i+2][0] - pts[i][0])
			times.append(pts[i][0])

		return times, periods

	def calculate_APD_X(self, **kwargs):
		times, apds, dis = self.calculate_APD_X_with_DI(**kwargs)
		return times, apds

	def calculate_APD_by_percentage(self, **kwargs):
		times, apds, dis = self.calculate_APD_by_percentage_with_DI(**kwargs)
		return times, apds

	def calculate_APD_X_with_DI(self, **kwargs):
		self.load_udat()

		kwargs['scale'] = self.dT
		kwargs['data'] = list(self.udat[:,kwargs['x']])[kwargs['start']:]
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

		peak = np.max(list(self.udat[:,kwargs['x']])[kwargs['start']:])
		min_value = np.min(self.udat[kwargs['start']:])
		APD_level = peak - (peak - min_value)*APD_percent
		print("Max is %.2f, min is %.2f, APD_level %.2f" % (peak, min_value, APD_level))

		kwargs['u_level'] = APD_level
		return self.calculate_APD_X_with_DI(**kwargs)

	def get_velocity(self, **kwargs):
		# left, right, u_level

		self.load_udat()

		point_l, point_r = (kwargs['left'], kwargs['right'])

		point_l_ind = int(point_l/(self.mult*self.dr))
		point_r_ind = int(point_r/(self.mult*self.dr))
		exact_point_l=point_l_ind*self.mult*self.dr
		exact_point_r=point_r_ind*self.mult*self.dr

		kwargs['scale'] = self.dT
		kwargs['data'] = self.udat[:,point_l_ind]
		asc, points_l = self.preprocess_calculation(**kwargs)
		kwargs['data'] = self.udat[:,point_r_ind]
		asc, points_r = self.preprocess_calculation(**kwargs)

		points_l = points_l[::2]
		points_r = points_r[::2]

		zipped = list(zip(points_l, points_r))
		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			print("Exact points L: %.2f, R: %.2f" % (exact_point_l, exact_point_r))
			print("Time at L: %.2f and R: %.2f" % (zipped[-1][0][0], zipped[-1][1][0]))
			print("Length %.2f" % float(point_r - point_l))
			print("Time %.2f" % (zipped[-1][1][0] - zipped[-1][0][0]))

		return (exact_point_r - exact_point_l)/(zipped[-1][1][0] - zipped[-1][0][0])

	def draw_heatmap(self, plt, **kwargs):
		self.load_udat()
		plt.yticks(range(0, self.time + 1, 500))
		ax = plt.gca()
		ax.yaxis.set_ticks_position('both')
		plt.tight_layout()
		plt.imshow(np.transpose(self.udat), origin='lower', interpolation='none', extent=[0, self.time, 0, self.N*self.dr], aspect='auto', 
			vmax=float(kwargs['vmax']) if 'vmax' in kwargs.keys() else 25)
		plt.xlabel('Time, ms', fontsize=14)
		plt.ylabel('X-coordinate, mm', fontsize=14)
		plt.tight_layout()
		plt.show()

	def get_data_vs_stimulated(self, x=50, last_count=10, start_t=0, u_level=-70, _error=.3, data_type='measured'):
		# data_type = ['measured', 'APD']
	    start_t_ind = int(start_t/self.dT)
	    x_ind = int(x/(self.mult*self.dr))
	    electrode_width = 2.5
	    norm_vel = 0.68
	    time_to_point = (x-electrode_width)/norm_vel
	    error = _error * self.period

	    if data_type == 'measured':
	    	times, periods = self.calculate_periods(x=x_ind, start=start_t_ind, u_level=u_level)
	    elif data_type == 'APD':
	    	times, periods = self.calculate_APD_by_percentage(x=x_ind, start=start_t_ind, u_level=u_level, APD_percent=.9)

	    stim_vs_msrd = []
	    res_stim = []
	    self.load_stimtable()
	    stim_table = self.stimtable
	    for t,p in zip(times, periods):
	        try:
	            stim = stim_table[(stim_table[:,1] < t - time_to_point + error) & (stim_table[:,1] > t - time_to_point - error)][0]
	            stim_vs_msrd.append([stim[0], stim[1], p])
	        except Exception:
	            print("An error occured with period %d" % p)
	    #     stim_vs_msrd.extend({'stim_period': stim[0][0], 'stim_moment': stim[0][1], 'measured_period': p})

	    stim_vs_msrd=np.array(stim_vs_msrd)
	    
	    for p in stim_table[:,0]:
	        res_stim.extend(stim_vs_msrd[stim_vs_msrd[:,0] == p][-last_count:])

	    res_stim = np.array(res_stim)
	    return res_stim

	def get_measured_vs_stimulated(self, **kwargs):
		return self.get_data_vs_stimulated(**kwargs)