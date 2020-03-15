import pandas as pd
import configparser
import os, re
import numpy as np

class LVC_table:

    def __init__(self, results_path='', database_path='', path_template=''):
        self.results_path = results_path
        self.database_path = "%s/db.ini" % results_path if database_path == '' else database_path
        self.path_template = "%s/id_%%d" % results_path if path_template == '' else path_template
        self.settings = configparser.ConfigParser()
        self.settings.optionxform = lambda option: option # enabling case sensetivity
        self.settings.read(self.database_path)
        
    def get_tableDF(self, allowed=[]):
        return pd.DataFrame(self.settings2DataFrame(), columns=allowed)
        
    def settings2DataFrame(self):
        db = []
        settings = self.settings
        for exp in settings:
            vrs = settings[exp]
            if exp == 'main' or exp == 'DEFAULT':
                continue
            id = int(exp)
            v_arr = {}
            for option in vrs:
                    v_arr[option] = vrs[option]
            v_arr['path'] = self.path_template % id
            v_arr['status'] = checkStatus(v_arr['path'])
            db.append(v_arr)

        return db
                
    
def checkStatus(case):
    dirs = os.listdir(case)
    if len(list([x for x in dirs if re.match('slurm', x) is not None])) > 0:
        dirs.append('') # add current dir
    dirs = [x for x in dirs if x != 'source' and x != 'sources' and os.path.isdir("%s/%s" % (case, x))]
    STATUS = "EMPTY"
    for dr in dirs:
        files = os.listdir("%s/%s" % (case, dr))
        slurm_files = [x for x in files if re.match('slurm', x) is not None]
        
        if len(slurm_files) == 0:
            STATUS = "PENDING"
            break
        else:
            slurm_content = open("%s/%s/%s" % (case, dr, slurm_files[-1]), 'r').read()
            if 'Time of calculation' in slurm_content:
                STATUS = "COMPLETED"
            elif 'Exited with exit code 2' in slurm_content or 'CANCELLED' in slurm_content:
                STATUS = "FAILED"
                break;
            else:
                proc_tab = np.array(re.findall(r'(\d+).\d+%', slurm_content))
                proc_tab = proc_tab.astype(np.int)
                if len(proc_tab) > 0:
                    STATUS = "RUNNING (%d%%)" % proc_tab[-1]
                else:
                    STATUS = "None"
    
    return STATUS