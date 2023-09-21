import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

class rhessys_calibrater:
    
    def __init__(self, work_dir, para_dic=None):
        # Work_dir: path of the rhessys model
        if para_dic is None:
            self.para_dic = {'gw1':(0.001,0.2),
                             'gw2':(0.001,0.2),
                             'gw3':(0.001,0.2),
                             's1':(0.001,20),
                             's2':(0.1,300.0),
                             's3':(0.1,20),
                             'sv1':(0.001,20),
                             'sv2':(0.1,300.0),
                             'svalt1':(0.5,2),
                             'svalt2':(0.5,2),
                            # 'snowEs ':(0.5,2),
                            # 'snowTs ':(0.5,2)
                               }
            self.workdir = work_dir
            
            if not os.path.exists(self.workdir):
                raise Exception(f'The RHESSys folder {self.workdir} does not exist.')
            
            print(f'RHESSys calibrater is setup at {self.workdir}')
            
            if not os.path.exists(os.path.join(self.workdir, 'output_calibration')):
                os.mkdir(os.path.join(self.workdir, 'output_calibration'))
                print('"output_calibration" is created (to store RHESSys calibration output)')
            else:
                print("output_calibration folder already exists.")
            
            if not os.path.exists(os.path.join(self.workdir, 'jobfiles')):
                os.mkdir(os.path.join(self.workdir, 'jobfiles'))
                print('"jobfiles" is created (to store temporary job files)')
            else:
                print("jobfiles folder already exists.")
            
            self.output_folder = 'output_calibration'
            self.jobfiles_folder = 'jobfiles'
            
            
            # print(f'where two folders are created:\n\n"jobfiles": to store temporary job files')
    
    def UniformSample(self,
                      out_filename, 
                      num_set=1000, 
                      random_seed=2023):
        ##########
        ## Sample from uniform distribution for each parameter, mainly used for the
        ##     first round of RHESSys calibration
        ## Inputs:
        ##     1. para_dic: dictionary of parameters, in format of {par_name: [min, max]}
        ##     2. out_path: output file name, saved to RHESSys model folder
        ##     3. num_set: number of sets of parameters, default=1000
        ##########
        # Optional: Set a random seed to make results reproducable
        np.random.seed(random_seed)
        
        # 1. Draw samples for each parameter and save to a DataFrame
        os.path.join(self.workdir, out_filename)
        pd.DataFrame({name:np.random.uniform(self.para_dic[name][0],self.para_dic[name][1],num_set).round(5) for name in self.para_dic.keys()}).to_csv(
                   os.path.join(self.workdir,out_filename), index_label='index')
        
        # 2. Save the csv file storing parameter sets for future use
        #  df.to_csv(out_filename, index_label='index')
    
    def Para_cmd(self,
                 path_para_file, 
                 out_txt=None):
        ##########
        ## Submit jobs to UVA Rivanna. Requires:
        ##     1. path_para_sample: path of the csv file with parameter sets
        ##     2. allocation: allocation name on Rivanna
        ## Return:
        ##     a list of lines for parameters
        ##########

        # Step 1: Read parameter file
        lines = []
        paras = pd.read_csv(os.path.join(self.workdir, path_para_file))

        # Step 2: Create job script file for each line in the parameter file
        for idx in range(len(paras)):
        # for idx in range(1):
            cmd_line = ''

            para_line = paras.loc[idx]

            para_name = []
            for i in range(len(para_line)):
                if para_line.index[i] == 'index':
                    continue

                if para_line.index[i][:-1] not in para_name:
                    cmd_line += '-{} {} '.format(para_line.index[i][:-1], para_line[i])
                    para_name.append(para_line.index[i][:-1])
                else:
                    cmd_line += str(para_line[i]) + ' '

            lines.append(cmd_line[:-1])
        
        # Steo 3: Optional. Write out a txt file for command lines of parameters
        if out_txt is not None:
            with open(os.path.join(self.workdir, out_txt), 'w') as f:
                f.writelines(i+'\n' for i in lines)
        return lines
    
    def JobScripts(self,para_cmd_line,
                        rheesys_excutable,
                        st_date,ed_date,
                        worldfile,
                        flowtable,   # Note: there are two flow tables
                        tecfile,
                        options,
                        output_dir,
                        output_prefix,
                        allocation='lband_group',
                        submit=False,
                        mem_size='10gb',
                        time='1-00:00:00'):
        ############
        ## Function to create the job file for RHESSys submission
        ##     ** Recommend to define work_dir as the model directory **
        ##     ** and define other variables in "relative" path ** 
        ##     Arguments:
        ##     1. para_cmd_line: the command lines (strings) for parameters only
        ##     2. job_files: folder path to store all job files
        ##     3. work_dir: work directory for the RHESSys project
        ##     4. rhessys_executable: full path of the rhessys executable
        ##     5. st_date, ed_date: start and end date of simulation, format as 'YEAR MONTH DAY HOUR'
        ##     6. worldfile: path of worldfile, recommend to save under work_dir
        ##     7. flowtable: path of TWO flow tables, sub and surf flow
        ##     8. tecfile: tecfile path
        ##     9. options: options/flags for RHESSys (e.g., -g for growth mode, -b for basin output)
        ##     10. output_dir: folder for saving output files
        ##     11. output_prefix: prefix of output files
        ##
        ############
        
       #  cur_dir = os.getcwd()
        for idx, line in enumerate(para_cmd_line):
            # Set the idx starts at 1, not 0
            idx = idx + 1
            job_path = f'{self.workdir}/jobfiles/job_{idx}.job'
            with open(job_path,'w') as f:
                f.writelines('#!/bin/bash\n')
                f.writelines('#SBATCH --partition=standard\n')
                f.writelines('#SBATCH --ntasks=1\n')
                f.writelines(f'#SBATCH --mem={mem_size}\n')
                f.writelines(f'#SBATCH -t {time}\n')   # Change if simulation takes more than 10 hours
                f.writelines('#SBATCH --output=/dev/null\n')
                f.writelines('#SBATCH --error=/dev/null\n')
                f.writelines(f'#SBATCH --job-name=rs{idx}\n')
                f.writelines(f'#SBATCH -A {allocation}\n')
                # f.writelines(f'#SBATCH --mail-user={computing_id}@virginia.edu\n')
                # f.writelines(f'#SBATCH --mail-type=END\n')

                f.writelines('module purge\n')
                f.writelines('module load singularity\n')

                f.writelines(f'cd {self.workdir}\n')


                f.writelines(f'{rheesys_excutable} -st {st_date} -ed {ed_date} {options} '+
                            f'-t {tecfile} -w {worldfile} -whdr {worldfile}.hdr ' +
                            f'-r {flowtable} -pre {output_dir}/{output_prefix}{idx} ')

                f.writelines(line)
            
            if submit:
                os.system(f"sbatch {job_path}")
                print(f'Simulation {idx} is submitted.', end='\r')