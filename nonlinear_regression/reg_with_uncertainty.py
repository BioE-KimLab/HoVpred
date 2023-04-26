
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
np.random.seed(0)
from tqdm import tqdm

from contextlib import closing
from multiprocessing import Pool

def f(t, loga, c, b):
    # log HoV = log A + C log (B - T)
    return np.exp(loga + c * np.log(np.clip(b - t, 1E-8, np.inf)))

def kl_div_normal(mu1, mu2, sigma1, sigma2):
    return np.log(sigma2/sigma1) + ( ( sigma1 ** 2.0 + (mu1-mu2) ** 2.0 )  /   (2 * (sigma2**2.0))  ) - 0.5

def nonlinear_reg( sub_group ):
    smi, sub_df = sub_group 

    #if smi != 'c1ccc2c(c1)Cc1c3c(c4c(c1-2)Cc1ccccc1-4)-c1ccccc1C3': continue
    print(smi)

    sub_df = sub_df.sort_values(by=['temperature'], ascending=False)
    #print(sub_df.temperature)
    #print(sub_df['HoV (kJ/mol)'])


    ## deterministic nonlinear reg - not considering uncertainty
    p_opt, p_cov = curve_fit(f, sub_df['temperature'], sub_df['HoV (kJ/mol)'], p0 = [1., 1., 1600], maxfev = 1000)
    loga_det, c_det, b_det = p_opt

    hov_est_det = f(sub_df['temperature'], loga_det, c_det, b_det)
    mae_det = np.round(np.mean(np.abs(hov_est_det - sub_df['HoV (kJ/mol)'])), 2)

    #######################################################

    sampled_HoVs_allT = []
    KL_DIV = []
    for _, row in sub_df.iterrows():
        #sampled_HoVs = np.sort(np.random.normal(row['HoV (kJ/mol)'], row['Error'], 100))
        #sampled_HoVs = np.random.normal(row['HoV (kJ/mol)'], row['Error'], 100)
        sampled_HoVs = np.clip(  np.random.normal(row['HoV (kJ/mol)'], row['Error'], 100), a_min = 0,  a_max = 100000) 
        sampled_HoVs_allT.append(sampled_HoVs)

        kl_div_db_sample = kl_div_normal(row['HoV (kJ/mol)'], np.mean(sampled_HoVs), row['Error'], np.std(sampled_HoVs, ddof=1) )
        KL_DIV.append(kl_div_db_sample)
        
        '''
        if abs(row['HoV (kJ/mol)'] - np.mean(sampled_HoVs) ) > 1.0 or abs(row['Error'] - np.std(sampled_HoVs, ddof=1) ) > 1.0:
            print(row['HoV (kJ/mol)'], row['Error'])
            print( np.mean(sampled_HoVs), np.std(sampled_HoVs, ddof=1)  )
            print(smi)
        '''

    HoV_arrays_for_reg = np.array(sampled_HoVs_allT).T

    fail_count = 0
    LOG_A, C, B = [], [], []
    sampled_hov_diff = []
    for hov_one_trial in HoV_arrays_for_reg:
        try:
            p_opt, p_cov = curve_fit(f, sub_df['temperature'], hov_one_trial, p0 = [1., 1., 1600], maxfev = 1000 )
        except:
            fail_count += 1
            continue
        loga, c, b = p_opt

        sampled_hov_diff.append( np.mean( np.abs( hov_one_trial - sub_df['HoV (kJ/mol)']  ) ) )

        LOG_A.append(loga)
        C.append(c)
        B.append(b)

    hov_est = f(sub_df['temperature'], np.mean(LOG_A), np.mean(C), np.mean(B))
    mae = np.round(np.mean(np.abs(hov_est - sub_df['HoV (kJ/mol)'])), 2)

    '''
    return [np.mean(LOG_A), np.std(LOG_A, ddof=1),
            np.mean(C), np.std(C, ddof=1),
            np.mean(B), np.std(B, ddof=1),
            fail_count, len(sub_df), np.mean(KL_DIV), mae, np.mean(sampled_hov_diff),
            loga_det, c_det, b_det, mae_det]
    '''

    return {'smiles': smi, 'logA_stoch_mean': np.mean(LOG_A), 'logA_stoch_std': np.std(LOG_A, ddof=1),
            'C_stoch_mean': np.mean(C), 'C_stoch_std': np.std(C, ddof=1),
            'B_stoch_mean': np.mean(B), 'B_stoch_std': np.std(B, ddof=1),
            'fail_count': fail_count, 'n_data': len(sub_df), 'kl_div_two_normal_dists': np.mean(KL_DIV), 'MAE_stoch': mae, 'MAE_db_and_sampled_hov': np.mean(sampled_hov_diff),
            'logA_det': loga_det, 'C_det': c_det, 'B_det': b_det, 'MAE_det' :mae_det}

df = pd.read_csv('../data/Data_211005.csv')
#df = pd.read_csv('../data/tmp.csv')
subgroups = df.groupby('smiles')

p = Pool(4)
with closing(p):
    results = p.map(nonlinear_reg, subgroups)
    p.terminate()
    p.join()

result_df = pd.DataFrame(results)
result_df = result_df.round(
{'logA_stoch_mean': 3, 'logA_stoch_std': 3,
 'C_stoch_mean': 2, 'C_stoch_std': 2,
 'B_stoch_mean': 1, 'B_stoch_std': 1,
 'kl_div_two_normal_dists': 3, 'MAE_stoch': 2, 'MAE_db_and_sampled_hov': 2,
 'logA_det': 3, 'C_det': 2, 'B_det': 1, 'MAE_det' : 2}
)

result_df.to_csv('reg_with_unc_results.csv', index=False)

'''
for result in results:
    print(result)
'''
