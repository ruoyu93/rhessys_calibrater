from rhessys_calibrater_utils import *
import os

template = '/Users/royzhang/Documents/Github/rhessys_calibration/demo/param_template.csv'

calibrater = rhessys_calibrater(template=template)  # para_dict = para_dict if customized the ranges

#  2. Create parameter list as csv
calibrater.UniformSample()

# print(calibrater.lines) should be 0

#  3. Generate command lines based on the uniform sampling
#     Note: jobfiles will be saved to folder "jobfiles" under your work_dir

## Uncomment below on HPC
calibrater.JobScripts()