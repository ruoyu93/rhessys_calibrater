from rhessys_calibrater import *
import os

random_seed = 80092   # can be any number

parameter_df = 'example_params.csv'
set_num = 10   # Number of simualtions for calibration

computing_id = 'rz3jr'
# work_dir = f'/scratch/{computing_id}/RHESSys_Models/Nepal60m' # Change as needed
work_dir = os.path.join(os.getcwd(),'example')
rhessys_exe = '/home/rz3jr/RHESSysEastCoast-develop/rhessysEC.7.2'
st_date, ed_date = '1980 1 1 1', '2005 1 1 1'
worldfile = 'worldfiles/worldfile_crop'
flowtable = 'flows/surfflow.txt flows/subflow.txt'
tecfile = 'tecfiles/tec_daily_cali.txt'
output_dir = 'output_cali'
output_prefix = 'rhessys'
options = '-b -g -dynRtZoff -BGC_flag -gwtoriparian'

## Setup run time and memory for sbatch job
mem_size = '10gb'
time = '1-12:00:00'   # 1-day and 12 hours

# 1. Create the object for calibrating RHESSys
#    Note: This will create two folders, output_calibration & jobfiles, in your RHESSys project folder (i.e., work_dir)
#          Folder "output_calibration" stores all rhessys simulations
#                 "jobfiles" stores job files 
calibrater = rhessys_calibrater(work_dir=work_dir)

# 2. Create parameter list as csv
calibrater.UniformSample(parameter_df,num_set=10)

# 3. Generate command lines based on the uniform sampling
calibrater.JobScripts(calibrater.Para_cmd(parameter_df),
                        rhessys_exe,
                        st_date,ed_date,
                        worldfile,
                        flowtable,   # Note: there are two flow tables
                        tecfile,
                        options,
                        output_dir,
                        output_prefix,
                        mem_size=mem_size,
                        time=time)