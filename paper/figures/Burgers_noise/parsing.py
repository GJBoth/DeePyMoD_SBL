import numpy as np
from phimal_utilities.analysis import load_tensorboard
import pandas as pd
from os import listdir

def collect_runs(ID):
    files = [file for file in listdir('runs_correct/') if file[:len(ID)] == ID]#getting and sorting files
    files.sort()

    df_plot = pd.DataFrame() #df used to store results in
    noise = []
    run = []
    coeffs = []
    ini_coeffs = []
    ini_idx = []
    for file in files:
        df = load_tensorboard('runs_correct/' + file + '/')
        scaled_coeff_keys = [key for key in df.keys() if key[:6]=='scaled']
        coeffs.append(df.tail(1))
        noise.append(float(file.split('_')[1]))
        run.append(int(file.split('_')[3]))
        ini_sparse_idx = np.any(df[scaled_coeff_keys] == 0, axis=1).idxmax() - 1 # we want the step before
        ini_idx.append(ini_sparse_idx)
        ini_coeffs.append(df[scaled_coeff_keys].iloc[ini_sparse_idx].to_numpy())
        print(f'Done with {file}')
        
    df_plot['noise'] = noise
    df_plot['run'] = run
    df_plot['first_sparsity'] = ini_idx
    df_coeffs = pd.concat(coeffs).reset_index(drop=True)
    df_ini_coeffs = pd.DataFrame(np.stack(ini_coeffs, axis=0), columns = ['ini_' + key for key in scaled_coeff_keys])
    df_plot = pd.concat([df_plot, df_coeffs, df_ini_coeffs], axis=1)
    
    return df_plot

# Now run for all 
df = collect_runs('threshold')
df.to_pickle('data/threshold_collected.pd')

df = collect_runs('cluster')
df.to_pickle('data/cluster_collected.pd')

#df = collect_runs('pdefind')
#df.to_pickle('data/pdefind_collected.pd')