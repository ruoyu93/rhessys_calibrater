
from rhessys_calibrater import *
import os

#################################################################
#  This is an example to setup the initial/1st round calibration.
#  The package samples parameter sets from uniform distributions.
#################################################################


#################################################################
#  Setup required arguments to run calibration on a sbatch HPC
#################################################################
random_seed = 80092   # random seed for replicate output, can be any number

parameter_df = 'example_params_step1.csv'   # csv output file for sampled parameter sets
set_num = 1000                              # Number of simualtions for calibration, set large for 1st round (e.g. >=1000)


############
#  Arguments to setup job files on University of Virginia HPC, bash and slurm system
#  Note: currently only support SLURM job files
############
computing_id = 'rz3jr'                
# work_dir = f'/scratch/{computing_id}/RHESSys_Models/Nepal60m' # Change as needed
work_dir = os.path.join(os.getcwd(),'demo')
rhessys_exe = '/home/rz3jr/RHESSysEastCoast-develop/rhessysEC.7.2'
st_date, ed_date = '1980 1 1 1', '2005 1 1 1'
worldfile = 'worldfiles/worldfile_crop'
flowtable = 'flows/surfflow.txt flows/subflow.txt'
tecfile = 'tecfiles/tec_daily_cali.txt'
output_dir = 'output_cali'
output_prefix = 'rhessys'
options = '-b -g -dynRtZoff -BGC_flag -gwtoriparian'

mem_size = '10gb'     # memory for RHESSys run
time = '1-12:00:00'   # run time of RHESSys (e.g. 1-day and 12 hours)

## Optional: Customize parameter range for each parameter
## Physical meaning of each parameter can be found on wiki of https://github.com/RHESSys/RHESSys
## Below example is the dictionary for default ranges of gw, horizontal, and vertical soil profile

## Only change if you need customization. The ranges are saved in the calibrater by default
# para_dict = {'gw1':(0.001,0.2),
#              'gw2':(0.001,0.2),
#              'gw3':(0.001,0.2),
#              's1':(0.001,20),
#              's2':(0.1,300.0),
#              's3':(0.1,20),
#              'sv1':(0.001,20),
#              'sv2':(0.1,300.0),
#              'svalt1':(0.5,2),
#              'svalt2':(0.5,2),
#              # 'snowEs ':(0.5,2),
#              # 'snowTs ':(0.5,2)
#             }

##################################################################
#  Start to use the package to 
# -------------------------------------------
#  1. Create the object for calibrating RHESSys
#     Note: This will create two folders, output_calibration & jobfiles, in your RHESSys project folder (i.e., work_dir)
#           Folder "output_calibration" stores all rhessys simulations
#                  "jobfiles" stores job files 
calibrater = rhessys_calibrater(work_dir=work_dir)  # para_dict = para_dict if customized the ranges

#  2. Create parameter list as csv
calibrater.UniformSample(parameter_df,num_set=set_num)

#  3. Generate command lines based on the uniform sampling
#     Note: jobfiles will be saved to folder "jobfiles" under your work_dir

## Uncomment below on HPC
# calibrater.JobScripts(calibrater.Para_cmd(parameter_df),
#                         rhessys_exe,
#                         st_date,ed_date,
#                         worldfile,
#                         flowtable,   # Note: there are two flow tables
#                         tecfile,
#                         options,
#                         output_dir,
#                         output_prefix,
#                         mem_size=mem_size,
#                         time=time)