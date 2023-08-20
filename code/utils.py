import numpy as np
import json
import pandas as pd
import pathlib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import copy


FILE_PATH = str(pathlib.Path(__file__).parent.resolve())+'/'

def fold2ddG(fold,T=298):
    """takes a fold value (Kd_wt/Kd_mut) and make it a fold change
       ddG is in Kcal/mol
       R is 1.9872 cal/(K*mol)
       T is in kelvin
       """
    R = 1.9872/1000 #kcal
    return R*T*np.log(fold)

def ddg2fold(ddg,T=298):
    #reverse of fold2ddG
    R = 1.9872/1000 #kcal
    return np.exp(ddg/(R*T))

def fitAbRFC(x_train,y_train,clf_params,n_jobs=8):
    #fits a random forest classifier (AbRFC)
    #ytrain is the classification label (0=deleterious, 1=non-deleterious)
    #IterativeImputer is used for aif_score on non-ab/ag PPI
    #StandardScaler is not needed for tree-based models but it is used for consistency"""
    clf_params['n_jobs'] = n_jobs
    clf = make_pipeline(IterativeImputer(max_iter=10),StandardScaler(),RandomForestClassifier(**clf_params))
    clf.fit(x_train,y_train)
    return clf

def fitAbRFR(x_train,y_train,reg_params,n_jobs=8):
    #identical to fitAbRFC but for regression
    #ytrain is the ddG value or equivalent
    reg_params['n_jobs'] = n_jobs
    reg = make_pipeline(IterativeImputer(max_iter=10),StandardScaler(),RandomForestRegressor(**reg_params))
    reg.fit(x_train,y_train)
    return reg

def applyAbRFC(x_test,clf):
    #x_test is the test set with the same features as the training set
    #clf is the pipeline object that has been returned by fitAbRFC
    return clf.predict_proba(x_test)[:,1]

def applyAbRFR(x_test,reg):
    #x_test is the test set with the same features as the training set
    #reg is the pipeline object that has been returned by fitAbRFR
    return reg.predict(x_test)

def feature_importances(clf,features_keep,fold):
     ### Calculate the feature importances given a random forest classifier
     ##  values for all trees in the forest are used as to estimate the distribution.
     ##  graph is ranked based on median and may fluctuate slightly from run to run.
     fimps = []
     rf = clf.steps[2][1]
     std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
     for k in range(len(rf.feature_importances_)):
          fimps+=[{'feature':features_keep[k],'importance':np.round(rf.feature_importances_[k],3),
                              'importance_std':np.round(std[k],3), 'fold':fold}]
     return fimps

def run_cv():
    #runs the cross-validation for AbRFC, RF Regressor, AbLang, and GNN Classifier models
    #saves a csv with all the predicitions for all models and folds
    DATA_DIR = FILE_PATH+'../data/train/'
    CV = json.load(open(DATA_DIR+'CVIDS.json','r'))
    GNNPATH = DATA_DIR+'GNN_CV/'
    params = json.load(open(FILE_PATH+'/params.json'))

    CLF_PARAMS  =  copy.deepcopy(params['rf_params'])#parameters for RandomForestClassifier
    CLF_PARAMS['random_state'] = 19067 #set for reproducibility
    REG_PARAMS   = copy.deepcopy(CLF_PARAMS)
    del REG_PARAMS['class_weight'] #not used for regression
    del REG_PARAMS['criterion'] #not used for regression
    FOLD_CUTOFF = params['fold_cutoff'] #KD_WT/KD_MUT >= FOLD_CUTOFF --> non-deleterious
    DDG_CUTOFF = fold2ddG(FOLD_CUTOFF) #translate KD_WT/KD_MUT to ddG
    
    X = pd.read_csv(DATA_DIR+'Xtrain.csv')
    X['ID'] = X['ID'].astype(str)
    X.set_index('ID',inplace=True)
    X['aif_score'] =np.where(X['aif_score']=='False',np.NaN,X['aif_score']) #non-Ab/Ag PPI have no AIF score
    X['pdb_group'] = X['pdb_group'].astype(str)+X['chain']+X['res'] #groups to prevent data leakage
    X['y_clf'] = np.where(X['ddG']>=DDG_CUTOFF,1,0) #classification label (0=deleterious, 1=non-deleterious)
    
    all_fimps = [] #to store feature importances 
    all_data = []  #to store model predictions

    for FOLD in range(5):
        ix_train, ix_test = CV[str(FOLD)]     

        train_groups = X.loc[ix_train,'pdb_group'].unique()
        test_groups  = X.loc[ix_test,'pdb_group'].unique()
        assert len(set(train_groups).intersection(set(test_groups)))==0
        ######GET ABLANG DATA######
        #ablang is only used when the mutation is on the paratope
        abl = pd.read_csv(DATA_DIR+'ABLANG_CV/ablang_skempi.csv')
        abl['ablang_preds'] = abl['abl_pm']
        abl['ID'] = abl['ID'].astype(str)
        ix_test_abl =  list(set(ix_test).intersection(set(abl.ID.unique())))
        ypred_abl = abl.set_index('ID').loc[ix_test_abl,'ablang_preds'].values
        ###########################

        ######GET LABELS############
        ytrain = X.loc[ix_train,'y_clf'].values
        ytrain_reg  = X.loc[ix_train,'ddG'].values

        ytest = X.loc[ix_test,'y_clf'].values
        ytest_abl = X.loc[ix_test_abl,'y_clf'].values
        ytest_reg = X.loc[ix_test,'ddG'].values
        ############################

        #####AbRFC######
        xtrain_tree = X.loc[ix_train,params['features']].values
        clf = fitAbRFC(xtrain_tree,ytrain,CLF_PARAMS)

        #feature importances
        all_fimps+=feature_importances(clf,params['features'],FOLD)
        
        #predictions
        ypred = applyAbRFC(X.loc[ix_test,params['features']].values,clf)
        ####AbRFC END#####

        #####RF REGRESSOR####
        reg = fitAbRFR(xtrain_tree,ytrain_reg,REG_PARAMS)
        #predictions
        ypred_reg = applyAbRFR(X.loc[ix_test,params['features']].values,reg)
        ####REGRESSION END####

        ####GNN####
        gnn = pd.read_csv(GNNPATH+'cv_fold_{}_preds.csv'.format(FOLD))
        gnn['ID'] = gnn['ID'].astype(str)
        gnn_train = gnn.loc[gnn['is_train']==1,'ID'].values
        gnn_test  = gnn.loc[gnn['is_train']==0,'ID'].values

        #match means all ix_train in gnn_train AND no gnn_train in ix_test
        gnn_train = gnn.loc[gnn['is_train']==1,'ID'].values
        train_match = np.all(np.in1d(ix_train,gnn_train))
        val_match = ~np.any(np.in1d(gnn_train,ix_test))

        train_match = np.all(gnn.loc[gnn['is_train']==1,'ID'].apply(lambda x: x in ix_train)) and np.all([x in gnn.loc[gnn['is_train']==1,'ID'].values for x in ix_train])
        #assert val_match
        assert train_match and val_match
        #ypred_gnn_train = gnn.set_index('ID').loc[ix_train,'model_preds'].values
        ypred_gnn = gnn.set_index('ID').loc[ix_test,'GNN Classifier Score'].values
        ####GNN END####
        for i in range(len(ix_test)):
            d = {'FOLD':FOLD,'ID':ix_test[i],'y_reg':ytest_reg[i], 
                        'y_clf':ytest[i],'AbRFC':ypred[i],'RF Regressor':ypred_reg[i],
                        'GNN Classifier':ypred_gnn[i],'refAA':X.loc[ix_test[i],'refAA'],
                        'mutAA':X.loc[ix_test[i],'mutAA']}
            if ix_test[i] in ix_test_abl:
                ix_abl = ix_test_abl.index(ix_test[i])
                d['AbLang'] = ypred_abl[ix_abl]
            all_data+=[d]

    all_fimps = pd.DataFrame.from_records(all_fimps)
    fimp_file = DATA_DIR +'feature_importances.csv'
    print('Writing Feature Importances to {}'.format(fimp_file))
    all_fimps.to_csv(fimp_file,index=False)
    pred_file = DATA_DIR +'cv_preds.csv'
    print('Writing all data to {}'.format(pred_file))
    pd.DataFrame.from_records(all_data).to_csv(pred_file,index=False)

import re

FILE_PATH = '/Users/tbc/Aquene/ID/Corona/GCPaperFinal/GCNC/AbRFC/code/'
def numstr2tup(_n):
    match = re.match(r"([0-9]+)([a-z]+)", _n, re.I)
    if match:
        items = match.groups()
        return (int(items[0]),items[1])
    else:
        return (int(_n),' ')

def getCDR(_pos,cdrs,i=0):
    chain = _pos[0]
    resi,_ = numstr2tup(_pos[1:])
    for k,v in cdrs.items():
        if k[0]==chain:
            if resi>=v[i][0] and resi<=v[i][1]:
                return k
    return np.nan

def addSequenceLevel(scores,cdrs):
    scores['position'] = scores['chain']+scores['res']
    scores['cdr'] =  scores['position'].apply(lambda x: getCDR(x,cdrs))
    return scores

def pose2pdb(_pose,_rn):
    try:
        r_c = _pose.pdb_info().pose2pdb(_rn)
    except:
        return None
    
    _c = r_c.split()[1]
    _r = r_c.split()[0]+_pose.pdb_info().icode(_rn)
    
    return _r, _c

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
def addDsasa(scores,pdb):
    pyrosetta.init(extra_options="-packing:no_optH false -mute basic core")
    pose = pose_from_file(pdb)
    scorefxn = get_fa_scorefxn()
    scorefxn(pose)
    sides = ['LH','A']
    iam = InterfaceAnalyzerMover("{}_{}".format(sides[0],sides[1]))
    iam.set_compute_packstat(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    prd = iam.get_all_per_residue_data()
    ifeats_pr = {}
    dSASA = getattr(prd,'dSASA')
    dsasas = []
    for i in range(1,len(dSASA)+1):
        res,chain = pose2pdb(pose,i)
        res = res.strip()
        dsasas+=[{'chain':chain,'res':res,'dSASA':dSASA[i]}]

    dsasas = pd.DataFrame.from_records(dsasas)
    ol = len(scores)
    scores = pd.merge(scores,dsasas)
    if len(scores)!=ol:
        print(len(scores))
        print(ol)
        raise
    scores['dSASA_mut'] = scores['dSASA']-scores['idSASA_0_{}'.format(pdb.split('/')[-1].split('_')[0])]
    return scores

def addHumanFrequencies(scores):
    # ###ADD human frequencies
    l_freqs = pd.read_pickle(FILE_PATH+'../data/test/light_frequencies.pickle')
    h_freqs = pd.read_pickle(FILE_PATH+'../data/test/heavy_frequencies.pickle')

    l_freqs = np.round(l_freqs.divide(l_freqs['chain_count'],axis=0)*100,2)

    l_freqs = l_freqs[['D','M', 'C', 'L', 'E', 'H', 'R', 'N', 'S', 'V', 'P', 'K', 'I',
        'Y', 'F', 'G', 'W', 'T', 'Q', 'A']].stack()
    l_freqs = l_freqs.reset_index()
    l_freqs['chain']='L'
    l_freqs.columns = ['res','refAA','refAAHumanPCT','chain']

    h_freqs = np.round(h_freqs.divide(h_freqs['chain_count'],axis=0)*100,2)

    h_freqs = h_freqs[['D','M', 'C', 'L', 'E', 'H', 'R', 'N', 'S', 'V', 'P', 'K', 'I',
        'Y', 'F', 'G', 'W', 'T', 'Q', 'A']].stack()
    h_freqs = h_freqs.reset_index()
    h_freqs['chain']='H'
    h_freqs.columns = ['res','refAA','refAAHumanPCT','chain']
    freqs = pd.concat((l_freqs,h_freqs)).reset_index().drop('index',axis=1)
    ol = len(scores)
    scores = pd.merge(scores,freqs)
    if len(scores)!=ol:
        raise

    freqs = freqs.rename(columns={'refAA':'mutAA','refAAHumanPCT':'mutAAHumanPCT'})
    scores = pd.merge(scores,freqs)
    if len(scores)!=ol:
        raise
    return scores
    
def addCappedRank(_scores, _rank_col,max_per_position):
    per_position = {}
    n_capped = 0
    for i in range(_scores.loc[_scores[_rank_col]<99999,_rank_col].max()):
        ix = _scores.index[_scores[_rank_col]==i].values[0]
        pos = _scores.loc[ix,'chain']+_scores.loc[ix,'res']
        if pos in per_position: 
            if per_position[pos]>=max_per_position:
                pass
            else:
                per_position[pos]+=1
                _scores.loc[ix,'{} Capped'.format(_rank_col)] = n_capped
                n_capped+=1
        else:
            per_position[pos] = 1
            _scores.loc[ix,'{} Capped'.format(_rank_col)] = n_capped
            n_capped+=1
    _scores.loc[_scores['{} Capped'.format(_rank_col)].isnull(),'{} Capped'.format(_rank_col)] = 99999
    _scores['{}'.format(_rank_col)] = _scores['{} Capped'.format(_rank_col)].astype(int)
    _scores.drop('{} Capped'.format(_rank_col),axis=1,inplace=True)
    return _scores

def addRankings(scores,refAA_max,mutAA_min,per_position,mut_dsasa_only=False,noninterface_cdr_only=False):
    import scipy.stats
    ix_freq = (scores['refAAHumanPCT']<refAA_max)&(scores['mutAAHumanPCT']>=mutAA_min)
    #Interface Rank
    ix_if = ix_freq & ((scores['dSASA']!=0) | (scores['dSASA_mut']!=0))
    if mut_dsasa_only:
        ix_if = ix_freq & (scores['dSASA_mut']!=0)
    scores.loc[ix_if,'Interface Rank'] = np.sum(ix_if)-scipy.stats.rankdata(scores.loc[ix_if,'yclf'].values)
    scores['Interface Rank'].fillna(99999,inplace=True)
    scores['Interface Rank'] = scores['Interface Rank'].astype(int)
    scores = addCappedRank(scores,'Interface Rank',per_position)
    
    #Non-Interface Rank
    ix_if = ix_freq & ((scores['dSASA']==0)&(scores['dSASA_mut']==0))
    if noninterface_cdr_only:
         ix_if = ix_if&(scores['CDR'].isnull()==False)

    scores.loc[ix_if,'Non-Interface Rank'] = np.sum(ix_if)-scipy.stats.rankdata(scores.loc[ix_if,'yclf'].values)
    scores.loc[scores['Non-Interface Rank'].isnull(),'Non-Interface Rank'] = 99999
    scores['Non-Interface Rank'] = scores['Non-Interface Rank'].astype(int)
    scores = addCappedRank(scores,'Non-Interface Rank',per_position)
    return scores

def generateScores(params):
    abrfc_params = json.load(open(FILE_PATH+'params.json'))
    X = pd.read_csv(FILE_PATH+'../data/test/Xtrain_gc_used.csv')
    DDG_CUTOFF = fold2ddG(abrfc_params['fold_cutoff'])
    X['ID'] = X['ID'].astype(str)
    X.set_index('ID',inplace=True)
    X['aif_score'] =np.where(X['aif_score']=='False',np.NaN,X['aif_score'])
    X['y_clf'] = np.where(X['ddG']>=DDG_CUTOFF,1,0)
    CLF_PARAMS  = abrfc_params['rf_params']
    CLF_PARAMS['random_state'] = params['random_state'] #for reproducibility
    abrfc = fitAbRFC(X.loc[:,abrfc_params['features']].values,X.loc[:,'y_clf'].values,CLF_PARAMS)
    REG_PARAMS = copy.deepcopy(CLF_PARAMS)
    del REG_PARAMS['class_weight']
    del REG_PARAMS['criterion']
    abrfr = fitAbRFR(X.loc[:,abrfc_params['features']].values,X.loc[:,'ddG'].values,REG_PARAMS)
    
    # abrfc = pickle.load(open(params['abrfc'],'rb'))
    # abrfr = pickle.load(open(params['abrfr'],'rb'))
    pdb_specific_cols = ['yclf', 'yreg']+\
        ['idSASA_0','fa_sol_0','aif_score','sin_norm','dE2','fa_atr_0',
        'total_score_0','fa_elec_0','sin_res','idG_1']
    scores = None
    all_pdbs = []
    for PDB in params['feature_files'].keys():
        Xtest = pd.read_pickle(params['feature_files'][PDB])
        Xtest['yclf'] = abrfc.predict_proba(Xtest[abrfc_params['features']].values)[:,1]
        Xtest['yreg'] = abrfr.predict(Xtest[abrfc_params['features']].values)
        Xtest = Xtest.loc[:,['label','chain','res','refAA','mutAA']+pdb_specific_cols]
        Xtest.rename(columns={x:'{}_{}'.format(x,PDB) for x in pdb_specific_cols},inplace=True)
        if scores is None:
            scores = Xtest.copy(deep=True)
        else:
            scores = scores.merge(Xtest,on=['label','chain','res','refAA','mutAA'])
        all_pdbs+=[PDB]

    for col in pdb_specific_cols:
        cols_average = ['{}_{}'.format(col,x) for x in all_pdbs]
        scores[col] = scores[cols_average].mean(axis=1)
        if col not in ['idSASA_0']:
            scores.drop(cols_average,axis=1,inplace=True)

    scores = addSequenceLevel(scores,params['cdrs'])
    scores = addDsasa(scores,params['pdb'])
    #drop the idSASA_0 columns
    scores.drop(['{}_{}'.format('idSASA_0',x) for x in all_pdbs],axis=1,inplace=True)
    scores = addHumanFrequencies(scores)
    scores = addRankings(scores,params['refAA_max'], params['mutAA_min'],params['per_position'],noninterface_cdr_only=False)
    scores['model_selected'] = ((scores['Interface Rank']<=(params['n_interface']-1))|(scores['Non-Interface Rank']<=(params['n_noninterface']-1))).astype(int)
    if 'ablang_score' in Xtest.columns:
        scores = pd.merge(scores,Xtest[['label','ablang_score']])
    #scores.loc[scores.model_selected==0,'Interface Rank'] = 99999
    #scores.loc[scores.model_selected==0,'Non-Interface Rank'] = 99999
    scores = scores.rename(columns={'yclf':'AbRFC Score',
    'yreg':'RF Regressor Score'}).drop(['chain','mutAA','refAA','res'],axis=1)
    return scores
