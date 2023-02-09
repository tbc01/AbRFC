from code.utils import *
import json
import pandas as pd
import sys


if __name__ == '__main__':
    CLF_PARAMS = json.load(open('PARAMS.json','r'))
    X = pd.read_csv('data/X.csv')
    if len(sys.argv) > 1:
        run = sys.argv[1]

        features_keep = ['idSASA_fraction_1', 'idhSASA_sc_0', 'idSASA_0', 'dE2', 
                    'iinterface_residues_1', 'idSASA_fraction_0', 'idSASA_sc_0', 
                    'fa_rep_1', 'hbond_bb_sc_0', 'fa_atr_1', 'iinterface_hbonds', 'isc_value', 
                    'idelta_unsat_hbonds', 'fa_elec_0', 'sin_res', 'sin_norm', 'lk_ball_wtd_1', 
                    'fa_atr_0', 'total_score_0', 'fa_sol_0', 'hbond_bb_sc_1', 'total_score_1', 
                    'hbond_sc_0', 'idhSASA_sc_1', 'fa_elec_1', 'idG_1', 'fa_sol_1', 'aif_score',
                    'sin_if', 'idhSASA_1', 'fa_rep_0']

        if run=='cv':
            X = X[X['dataset']=='SKEMPI']
            X.set_index('ID',inplace=True)
            CV = json.load(open('data/CVIDS.json','r'))
            GNNPATH ='data/internal_data/'
            ABLANGFILE = 'data/ablang_skempi.csv'
            features_keep = features_keep = ['idSASA_fraction_1', 'idhSASA_sc_0', 'idSASA_0', 'dE2', 
                            'iinterface_residues_1', 'idSASA_fraction_0', 'idSASA_sc_0', 
                            'fa_rep_1', 'hbond_bb_sc_0', 'fa_atr_1', 'iinterface_hbonds', 'isc_value', 
                            'idelta_unsat_hbonds', 'fa_elec_0', 'sin_res', 'sin_norm', 'lk_ball_wtd_1', 
                            'fa_atr_0', 'total_score_0', 'fa_sol_0', 'hbond_bb_sc_1', 'total_score_1', 
                            'hbond_sc_0', 'idhSASA_sc_1', 'fa_elec_1', 'idG_1', 'fa_sol_1', 'aif_score',
                            'sin_if', 'idhSASA_1', 'fa_rep_0']
            X['aif_score'] =np.where(X['aif_score']=='False',np.NaN,X['aif_score'])
            crossval(features_keep,CLF_PARAMS,X,CV,GNNPATH,ABLANGFILE)
        
        if run=='score':
            X.set_index('ID',inplace=True)
            X['aif_score'] =np.where(X['aif_score']=='False',np.NaN,X['aif_score'])
            score_dataset(features_keep,CLF_PARAMS,X,'GMAB')
            score_dataset(features_keep,CLF_PARAMS,X,'CMAB')
    else:
        raise ValueError('Please specify a run type: cv or score')