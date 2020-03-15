import configparser
import os
import subprocess
import itertools
from shutil import copyfile, rmtree

class LVC_launcher:

    def __init__(self, recompile=True, source=True, cpus=12, additional_jobs=[], **kwargs):
        self.partitions = kwargs['partitions']
        self.need_recompile = recompile
        self.need_source = source
        self.cpus = cpus
        self.sim_time = kwargs['sim_time']
        self.source_list = ['AP.c', 'elektro.h', 'LR.c', 'makefile', 'main.c', 'TP06-tab.c']
        self.source_list.extend(kwargs['additional_source']) if 'additional_source' in kwargs.keys() else None
        self.template_file = kwargs['template']
        self.source_list.append(self.template_file)
        self.working_dir = kwargs['working_dir']
        self.experiments_folder = "%s/%s" % (self.working_dir, 
            kwargs['experiments_folder'] if 'experiments_folder' in kwargs.keys() else 'results')
        self.template_path = "%s/%s" % (self.working_dir, self.template_file)
        self.additional_jobs = additional_jobs
        self.EXP_ID = None
        
    def run(self, fake=False):
        rec_err = (self.recompile() if self.need_recompile else b'')
        if rec_err != b'':
            print("Failed to compile. Exiting ...")
            rm_exp(self.output_folder, self.EXP_ID)
            return None
        self.prepare_workspace()
        self.copy_source() if self.need_source else None
        self.gen_exp_folder()
        self.do_additional_job()
        self.update_db()
        if not fake:
            self.launch()
            
    def rerun(self, fake=False, exp_id=None, new_params={}, description=None, new_series=True):
        if not new_series:
            self.EXP_ID = exp_id
        rec_err = (self.recompile() if self.need_recompile else b'')
        if rec_err != b'':
            print("Failed to compile. Exiting ...")
            rm_exp(self.output_folder, self.EXP_ID)
            return None
        self.prepare_workspace()
        # read params from section
        self.parameters = dict(self.settings["%d" % exp_id])
        if description != None:
            self.description = description
        else:
            self.description = self.settings["%d" % exp_id]['Description']
        del self.parameters['Description']
        # parse params
        for key, value in self.parameters.items():
            self.parameters[key] = parse_param(key, value)
        # change new params
        for k,v in new_params.items():
            if type(v) != type([]):
                v = [v]
            self.parameters[k] = v        
        # clean folder and create new ones
        if not new_series:
            dirs = os.listdir(self.output_folder)
            dirs = ["%s/%s" % (self.output_folder, x) for x in dirs if x != 'source' and x != 'sources' and os.path.isdir("%s/%s" % (self.output_folder, x))]
            for dr in dirs:
                rmtree(dr)
        self.gen_exp_folder()
#         update db and launch
        self.do_additional_job()
        self.update_db()
        if not fake:
            self.launch()
                
        
    def prepare_workspace(self):
        if not os.path.exists(self.experiments_folder):
            os.makedirs(self.experiments_folder)

        self.settings_path = "%s/db.ini" % self.experiments_folder
        self.settings = configparser.ConfigParser()
        self.settings.optionxform = lambda option: option # enabling case sensetivity
        if os.path.exists(self.settings_path):
            self.settings.read(self.settings_path)
            if self.EXP_ID == None:
                self.EXP_ID = int(self.settings.sections()[-1]) + 1
        else:
            self.EXP_ID = 0

        self.output_folder = "%s/id_%d" % (self.experiments_folder , self.EXP_ID)
        
    def recompile(self):
        os.chdir(self.working_dir)
        p = subprocess.Popen("make -f %s/makefile" % self.working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = p.communicate()
        print("Compilation output:", output)
        print("Compilation errors:", error)
        return error
        
    def copy_source(self):
        os.chdir(self.working_dir)
        sources = "%s/sources" % self.output_folder
        if not os.path.exists(sources):
                os.makedirs(sources)
        for file in self.source_list:
            copyfile(file, "%s/%s" % (sources, file))

    def set_exp_parameters(self, parameters, description='no descr.'):
        self.parameters = {k : v if type(v) == type([]) else [v] for k,v in parameters.items()}
        self.description = description

    def gen_exp_folder(self):
        with open(self.template_path, 'r') as f:
            array = f.read().replace("\n","=").split("=")
            self.conf = {array[i] : array[i+1] for i in range(0, len(array) - 1, 2)}
        self.launch_strings = []
        varied = []
        dirname_template = self.output_folder + "/"
        for key in self.parameters.keys():
            varied.append([(key, p) for p in self.parameters[key]])
            if len(self.parameters[key]) > 1:
                dirname_template += "%s=%%s" % key
        tuples = list(itertools.product(*varied))
        for parameters_tuple in tuples:
            experiment_dir = dirname_template % tuple(v[1] for v in parameters_tuple if len(self.parameters[v[0]]) > 1)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            p = subprocess.Popen(['cp','-p','--preserve', "LVD", experiment_dir + "/LVD"])
            p.wait()

            for pair in parameters_tuple:
                self.conf[pair[0]] = pair[1]

            write_config(self.conf, experiment_dir + "/input.txt")
            self.launch_strings.append(experiment_dir)
    
    def do_additional_job(self):
        if len(self.additional_jobs) > 0:
            for experiment in self.launch_strings:
                os.chdir(experiment)
                for job in self.additional_jobs:
                    p = subprocess.Popen(job, shell=True)
                    p.communicate()
    
    def update_db(self):
        conf = self.conf
        if not self.settings.has_section('%d' % self.EXP_ID):
            self.settings.add_section('%d' % self.EXP_ID)
        for key in conf.keys():
            self.settings['%d' % self.EXP_ID][key] = str(conf[key])
        for key in self.parameters.keys():
            self.settings['%d' % self.EXP_ID][key] = ', '.join(map(str, self.parameters[key]))
            self.settings['%d' % self.EXP_ID]['Description'] = self.description
        with open(self.settings_path, 'w') as settings_file:
            self.settings.write(settings_file)
            
    def launch(self):
        for experiment in self.launch_strings:
            print("Launching %s" % experiment)
            os.chdir(experiment)
            p = subprocess.Popen("sbatch -N 1 -p %s --cpus-per-task=%d -t %s --wrap 'srun ./LVD'" % (self.partitions, self.cpus, self.sim_time), 
                                 shell=True, env={'LD_LIBRARY_PATH' : '/opt/intel/composer_xe_2013_sp1/lib/intel64/'})
            p.communicate()

def rm_exp(exps_path='', exp_id=-1):
    candidate_path = '%s/id_%d' % (exps_path, exp_id)
    db_path = "%s/db.ini" % exps_path
    if os.path.exists(candidate_path):
        rmtree(candidate_path, ignore_errors=False)
    if os.path.exists(db_path):
        settings = configparser.ConfigParser()
        settings.read(db_path)
        settings.remove_section('%d' % exp_id)
        with open(db_path, 'w') as settings_file:
            settings.write(settings_file)
        
def rm_last(exps_path):
    pass
            
def write_config(config, config_file):
    with open(config_file, 'w') as f:
        for key in config.keys():
            f.write("%s=%s\n" % (key, str(config[key])))
            
def parse_param(key, value):
    floats = {'dt', 'drx', 'dry', 'dr', 'Diffuz1', 'Diffuz2', 'Diffuz', 't_stim', 'deltaT', 'u_low', 'c_gNa', 'S1_width', 'S2_width', 'S_last_t'}
    ints = {'pace_start', 'pace_period', 'space_mult_x', 'space_mult_y', 'print_every', 't1', 'nx', 'ny', 'S2_x', 'S2_y', 'I_stim', 
            'hole_x', 'hole_y', 'hole_size_x', 'hole_size_y', 'electrode', 'assim', 'Nbegin',  'Ttarget', 'Nrepeat', 'read_from_file', 'smartstim', 'chirality'}
    
    value = value.split(',') if ',' in value else [value]
    
    return list([float(v) if key in floats else int(v) for v in value])