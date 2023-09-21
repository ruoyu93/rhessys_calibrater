import numpy as np
import pandas as pd
import os
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
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
        else:
            self.para_dic = para_dic
        
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

class rhessys_cluster:
    
    def __init__(self, 
                param_df,    # the generated csv file 
                sim_folder,
                sim_prefix,
                ):
        self.param = pd.read_csv(param_df)
        self.sim_folder = sim_folder
        self.sim_prefix = sim_prefix
        
        self.new_param = None
        self.new_lines = []
        print("{} sets of parameters are simulated".format(len(self.param)))

        output_len = len(list(set([i.split('_')[0] for i in os.listdir(self.sim_folder)])))

        if output_len != len(self.param):
            raise Exception('Number of simulations NOT match given parameter table.')
        else:
            print('Parameter analyzing is ready for:')
            self.param_names = [i for i in self.param.columns if i != 'index']
            print(', '.join(self.param_names)+'.',len(self.param_names), 'in total.')
    
    def objective_analysis(self, 
                           observation, 
                           date_field,
                           Q_field,
                           period,
                           stat_out=None,
                           obj_funs=['NSE','rmse','r2'],
                           cols = ['laiTREE', 'gw.storage']):

        # 1. Read observation data, with datetime as index
        observ = pd.read_csv(observation)
        observ.index = pd.to_datetime(observ[date_field])

        # 3. Create columns for each objective functions
        for name in obj_funs: self.param[name] = np.nan 
        for name in cols: self.param[name] = np.nan 

        for idx in range(1, len(self.param) + 1):
            try:
                pred = pd.read_csv(f'{self.sim_folder}/{self.sim_prefix}{idx}_basin.daily', sep=' ')  # use basin daily file to calibrate
            except:
                print(f'Simulation {idx} failed.')
                continue
            
            if len(pred) == 0:
                print(f'Simulation {idx} was corrupted. Skipped...')
            else:
                print(f'Analyzing {idx}', end="\r", flush=True)
                pred.index = pd.to_datetime(pred['year'].astype(str)+'-'+pred['month'].astype(str)+'-'+pred['day'].astype(str),format='%Y-%m-%d')
                
                if pred.index[-1] < pd.to_datetime(period[1]):
                    Q_pred = pred['streamflow'][period[0]:pred.index[-1]]
                    Q_obs = observ[Q_field][period[0]:pred.index[-1]]
                    print(f'Warning: Simulation {idx} ends earlier than the defined ending date')
                    print('Analyzing to {}-{}-{} only'.format(pred.index[-1].year,pred.index[-1].month,pred.index[-1].day))
                else:
                    Q_pred = pred['streamflow'][period[0]:period[1]]
                    Q_obs = observ[Q_field][period[0]:period[1]]
                    
                for fun in obj_funs:
                    self.param[fun][idx-1] = getattr(rhessys_cluster, fun)(Q_obs, Q_pred)
                for col in cols:
                    # print(f'  Calculating col {col}', end="\r", flush=True)
                    self.param[col][idx-1] = np.mean(pred[col][period[0]:period[1]])
        
        if stat_out is not None:
            self.param.to_csv(stat_out, index=False)
        return self.param
    
    def KclusterSampling(self, 
                obj_fun,
                threshold, 
                table_out,
                method='greater',
                sample_size=50,  # number of sampling from each cluster
                kmeans_kwargs = {'init': 'random', 
                                 'n_init':10, 
                                 'max_iter':300, 
                                 'random_state':888}):

        if method == 'greater':
            temp = self.param[self.param[obj_fun] >= threshold][self.param_names]
        else:
            temp = self.param[self.param[obj_fun] <= threshold][self.param_names]

        if len(temp) < 0.1*len(self.param):
            print("Threshold is too strict to perform K clustering with only {} points".format(len(temp)))
            if method == 'greater':
                sorting = False
            else:
                sorting = True
            
            temp = self.param.sort_values(by=obj_fun, ascending=sorting).iloc[:int(0.1*len(self.param)),:][self.param_names]
            print("Increasing sample to 10% of the simulations {}/{}".format(int(0.1*len(self.param)),len(self.param)))
        
        # Perform K clustering analysis
        silhouette_coef = []

        for k in range(2,8):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(temp)
            silhouette_coef.append(silhouette_score(temp, kmeans.labels_))
        
        opt_nodes = np.argmax(silhouette_coef)+2
        if opt_nodes <= 3:
            opt_nodes = 4
        
        print(f"Found {opt_nodes} clusters ...")
        
        kmeans = KMeans(n_clusters=opt_nodes, **kmeans_kwargs)

        df_list = []

        temp['kmean_group'] = kmeans.fit(temp).labels_
        cluster_mean = temp.groupby('kmean_group').mean()
        cluster_std = temp.groupby('kmean_group').std()
        for i in range(len(cluster_mean)):
            df=pd.DataFrame()
            # print(i)
            for col in self.param_names:
                df[col] = np.random.normal(loc=cluster_mean[col][i], 
                                           scale=cluster_std[col][i], 
                                           size=sample_size)
            df_list.append(df)
        new_param = pd.concat(df_list).reset_index(drop=True)
        new_param['index'] = new_param.index + 1

        new_param.to_csv(table_out, index=False)
        self.new_param = new_param
        return new_param
    
    def Jobfiles_newcali(self,
                       work_dir,
                       rheesys_excutable,
                       st_date,ed_date,
                        worldfile,
                        flowtable,   # Note: there are two flow tables
                        tecfile,
                        options,
                        output_dir,
                        output_prefix,
                         submit='False',
                       allocation='lbandgroup',
                       mem_size='10gb',
                       time='1-10:00:00'):
        ##########
        ## Submit jobs to UVA Rivanna. Requires:
        ##     1. path_para_sample: path of the csv file with parameter sets
        ##     2. allocation: allocation name on Rivanna
        ## Return:
        ##     a list of lines for parameters
        ##########
        
        if len(self.new_param) < 1:
            raise Exception('No new parameters generated, run ".KclusterSampling()" first.')
        
        # Step 1: Read parameter file
        paras = self.new_param

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

            self.new_lines.append(cmd_line[:-1])
            
            for idx, line in enumerate(self.new_lines):
            # Set the idx starts at 1, not 0
                idx = idx + 1
                job_path = f'{work_dir}/jobfiles/job_{idx}.job'
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

                    f.writelines(f'cd {work_dir}\n')


                    f.writelines(f'{rheesys_excutable} -st {st_date} -ed {ed_date} {options} '+
                                f'-t {tecfile} -w {worldfile} -whdr {worldfile}.hdr ' +
                                f'-r {flowtable} -pre {output_dir}/{output_prefix}{idx} ')

                    f.writelines(line)
            
                if submit: os.system(f"sbatch {job_path}")
                    # print(f'Simulation {idx} is submitted.', end='\r')

    def NSE(obs, pred):
        return 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
    
    def rmse(obs, pred):
        return np.mean(np.sqrt((obs - pred)**2))

    def r2(obs, pred):
        return np.corrcoef(obs, pred)[0,1]**2
        

