import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import seaborn as sns
import json
import copy


FILE_PATH = str(pathlib.Path(__file__).parent.resolve())+'/'
sys.path.append('{}/code'.format(FILE_PATH))

def figure2():
    def get_aa_counts(X):
        all_cols = 'ADEFGHIKLMNPQRSTVWY'
        tmp = X.groupby(['refAA','mutAA'])['ID'].count().unstack('mutAA').fillna(0)
        for c in all_cols:
            if c not in tmp.columns:
                tmp[c] = 0
            if c not in tmp.index:
                tmp.loc[c] = 0
        tmp = tmp.loc[list(all_cols),list(all_cols)]
        return tmp.astype(int)
    
    Xtrain = pd.read_csv(FILE_PATH+'data/train/Xtrain.csv')
    Xtrain = Xtrain[Xtrain['ID'].astype(int)<Xtrain['pair_ID'].astype(int)] #remove the augmented datapoints
    Xval = pd.read_csv(FILE_PATH+'data/validation/Xval.csv')
    Xval = Xval.rename(columns={'label':'ID'})
    Xtest = pd.read_csv(FILE_PATH+'data/validation/Xval_sm.csv')#saturation mutagenesis used as the simulation
    Xtest = Xtest.rename(columns={'label':'ID'})
    train_dist = get_aa_counts(Xtrain)
    val_dist = get_aa_counts(Xval)
    test_dist = get_aa_counts(Xtest)
    fig = plt.figure(figsize=(14,6.53*2))
    gs = fig.add_gridspec(2,2)
    axc = fig.add_subplot(gs[0, 0])
    axa = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 0])
    sns.heatmap(train_dist,annot=True,cmap='Blues',
                linewidths=1,linecolor='black',fmt='g',cbar=False,ax=axa)
    sns.heatmap(val_dist,annot=True,cmap='Blues',
                linewidths=1,linecolor='black',fmt='g',cbar=False,ax=axb)
    sns.heatmap(test_dist,annot=True,cmap='Blues',
                linewidths=1,linecolor='black',fmt='g',cbar=False,ax=axc)
    axb.set_xlabel('Mutated AA',fontsize=16)
    axb.set_ylabel('Reference AA',fontsize=16)
    axb.set_title('Validation Set Mutation Frequencies',fontsize=18)
    axb.set_xticklabels(axb.get_xticklabels(),fontsize=16)
    axb.set_yticklabels(axb.get_yticklabels(),fontsize=16)

    axa.set_xlabel('Mutated AA',fontsize=16)
    axa.set_ylabel('Reference AA',fontsize=16)
    axa.set_title('Training Set Mutation Frequencies',fontsize=18)
    axa.set_xticklabels(axb.get_xticklabels(),fontsize=16)
    axa.set_yticklabels(axb.get_yticklabels(),fontsize=16)

    axc.set_xlabel('Mutated AA',fontsize=16)
    axc.set_ylabel('Reference AA',fontsize=16)
    axc.set_title('Mutation Frequencies During Saturation Mutagenesis',fontsize=18)
    axc.set_xticklabels(axb.get_xticklabels(),fontsize=16)
    axc.set_yticklabels(axb.get_yticklabels(),fontsize=16)
    plt.tight_layout()
    plt.savefig(FILE_PATH+'outputs/Fig2.png')

def figures3CS1():
    data = pd.read_csv(FILE_PATH+'data/train/cv_preds.csv')
    models = ['AbRFC', 'RF Regressor', 'GNN Classifier', 'AbLang']
    fig, ax = plt.subplots(3,2,figsize=(7,7))
    results = []
    for FOLD in range(5):
        pltx = FOLD%2
        plty = FOLD//2
        for model in models:
            data_fold = data[(data['FOLD'] == FOLD)]
            if model == 'AbLang':
                data_fold = data[(data['FOLD'] == FOLD)&(data['AbLang'].isnull()==False)]
            

            ytrue = data_fold['y_clf']
            ypred = data_fold[model]

            precision, recall, _ = precision_recall_curve(ytrue, ypred)
            results += [{'FOLD': FOLD, 'Model': model, 'PR AUC':np.round(auc(recall, precision),2)}]
            ax[plty,pltx].plot(recall,precision,label=model)
        random_precision = data[(data['FOLD'] == FOLD)]['y_clf'].sum()/len(data[(data['FOLD'] == FOLD)]['y_clf'])
        ax[plty,pltx].plot([0,1],[random_precision,random_precision],'k--',label='Baseline')
        ax[plty,pltx].set_ylim(.5,1.05)
        ax[plty,pltx].set_ylim(.5,1.05)
        ax[plty,pltx].set_title('Fold %s '%FOLD,fontsize=12)
        ax[plty,pltx].set_xlabel('Recall',fontsize=12)
        ax[plty,pltx].set_ylabel('Precision',fontsize=12)
    results = pd.DataFrame.from_records(results)
    results = results.groupby(['FOLD','Model']).mean().unstack('Model')
    results.columns = results.columns.droplevel(0)
    results = results.round(2)
    fresults = FILE_PATH+'outputs/Fig3C.xlsx'
    results.loc['Average',:] = results.mean()
    results = results.round(2)
    results.to_excel(fresults)
    plt.savefig(FILE_PATH+'outputs/FigS1.png')

def figure3A():
    fig,axa = plt.subplots(1,1,figsize=(7,6.56))
    all_fimps = pd.read_csv(FILE_PATH+'data/train/feature_importances.csv')
    all_fimps = pd.merge(all_fimps,all_fimps.groupby('feature')['importance'].median().reset_index().rename(columns={'importance':'median_importance'}))
    sns.boxplot(data=all_fimps.sort_values('median_importance',ascending=False),x='feature',y='importance',ax=axa)
    axa.set_xticklabels(axa.get_xticklabels(),rotation=50,ha='right',rotation_mode='anchor',fontsize=10)
    axa.set_yticklabels([np.round(x,2) for x in axa.get_yticks()],fontsize=12)

    axa.yaxis.grid()
    axa.set_title('Random Forest Classifier Feature Importance\n(Cross Validation)',fontsize=14)
    axa.set_ylabel('Mean Decrease in\n Gini Impurity',fontsize=14)
    axa.set_xlabel('')
    plt.tight_layout()
    plt.savefig(FILE_PATH+'outputs/Fig3A.png')

def figure3B():
    from utils import fitAbRFC, applyAbRFC, fold2ddG
    ##parameter setup
    ##see code/utils/run_cv method for details
    params = json.load(open(FILE_PATH+'code/params.json'))
    HYPER_PARAMS  = copy.deepcopy(params['rf_params'])#json.load(open('./GCPaperFinal/AbRFC/params.json'))['rf_params']
    HYPER_PARAMS['n_jobs'] = 8
    HYPER_PARAMS['min_samples_split'] = 2
    HYPER_PARAMS['n_estimators'] = 1000
    HYPER_PARAMS['random_state'] = 19067
    CV = json.load(open(FILE_PATH+'data/train/CVIDS.json','r'))
    X = pd.read_csv(FILE_PATH+'data/train/Xtrain.csv')
    X['ID'] = X['ID'].astype(str)
    X.set_index('ID',inplace=True)
    X['aif_score'] =np.where(X['aif_score']=='False',np.NaN,X['aif_score']) #non-Ab/Ag PPI have no AIF score
    FOLD_CUTOFF = params['fold_cutoff'] #KD_WT/KD_MUT >= FOLD_CUTOFF --> non-deleterious
    DDG_CUTOFF = fold2ddG(FOLD_CUTOFF) #translate KD_WT/KD_MUT to ddG
    X['y_clf'] = np.where(X['ddG']>=DDG_CUTOFF,1,0) #classification label (0=deleterious, 1=non-deleterious)
    
    min_samples_leaf = []
    all_aucs_train = []
    all_aucs_test  = []


    for msl in [1,5,10,20,50,100,200,500]:
        pr_aucs_train = []
        pr_aucs_test  = []
        for FOLD in range(5):

            pltx = FOLD%2
            plty = FOLD//2
            
            ix_train, ix_test = CV[str(FOLD)]
            
            ytest = X.loc[ix_test,'y_clf'].values
            ytrain = X.loc[ix_train,'y_clf'].values

            #####TREE######
            xtrain_tree = X.loc[ix_train,params['features']].values
            HYPER_PARAMS['min_samples_leaf'] = msl
            clf = fitAbRFC(xtrain_tree,ytrain,HYPER_PARAMS)

            #tree preds
            ypred = applyAbRFC(X.loc[ix_test,params['features']].values,clf)
            ypred_train = applyAbRFC(xtrain_tree,clf)
            ####TREE END####

            
            ####PR CURVES####
            ptest, rtest, ttest = precision_recall_curve(ytest,ypred)
            ptrain, rtrain, ttrain = precision_recall_curve(ytrain,ypred_train)

            pr_aucs_test += [auc(rtest,ptest)]
            pr_aucs_train+= [auc(rtrain,ptrain)]

        min_samples_leaf += [msl]
        all_aucs_train += [np.mean(pr_aucs_train)]
        all_aucs_test  += [np.mean(pr_aucs_test)]
        print(msl,np.mean(pr_aucs_train),np.mean(pr_aucs_test))


    fig, ax = plt.subplots(1,1,figsize=(14/2,6.56*2/2))
    ax.plot(min_samples_leaf,all_aucs_train,'-x',label='Train Folds')
    ax.plot(min_samples_leaf,all_aucs_test,'-x',label='Validation Fold')
    ax.set_xlabel('Min Samples Leaf',fontsize=14)
    ax.set_ylabel('Precision Recall AUC',fontsize=14)
    ax.legend(fontsize=12)
    ax.set_title('Hyper Parameter Tuning\n(Cross Validation)',fontsize=14)
    ax.grid()
    ax.axvline(10,ls='--',color='g')
    ax.set_xticklabels([int(x) for x in ax.get_xticks()],fontsize=12)
    plt.tight_layout()
    plt.savefig(FILE_PATH+"outputs/Fig3B.pdf", dpi=300, format="pdf")

def figure3D():
    params = json.load(open(FILE_PATH+'code/params.json'))
    data = pd.read_csv(FILE_PATH+'data/train/cv_preds.csv')
    models = ['AbRFC', 'RF Regressor', 'GNN Classifier', 'AbLang']
    aa_bias = []
    baseline_bias = []
    n_total = 0
    baseline_n = 0
    for fold in data['FOLD'].unique():
        for model in models:
            data_fold = data[(data['FOLD'] == fold)&(data['AbLang'].isnull()==False)]
            top_n = int(np.ceil(len(data_fold)*.1))
            bias = data_fold.sort_values(by=model,ascending=False).iloc[:top_n]['mutAA'].value_counts().to_dict()
            for aa in bias:
                aa_bias+=[{'FOLD':fold,'model':model, 'AA':aa,'frequency':bias[aa]}]
            
        n_total += top_n
        data_fold = data[(data['FOLD'] == fold)&(data['AbLang'].isnull()==False)]
        bias = data_fold['mutAA'].value_counts().to_dict()
        for aa in bias:
            baseline_bias+=[{'FOLD':fold,'model':'Training Set Frequency','AA':aa,'frequency':bias[aa]}]

        baseline_n += len(data_fold)

    aa_bias = pd.DataFrame.from_records(aa_bias)
    aa_bias = aa_bias.groupby(['model','AA'])['frequency'].sum().reset_index()#.to_csv('./GCPaperFinal/GCNC/aa_bias.csv')
    aa_bias['frequency'] = aa_bias['frequency']/n_total

    baseline_bias = pd.DataFrame.from_records(baseline_bias)
    baseline_bias = baseline_bias.groupby(['model','AA'])['frequency'].sum().reset_index()#.to_csv('./GCPaperFinal/GCNC/aa_bias.csv')
    baseline_bias['frequency'] = baseline_bias['frequency']/baseline_n

    aa_bias = pd.concat([aa_bias,baseline_bias]).reset_index(drop=True)

    plot_order = models + ['Training Set Frequency']
    aa_bias['order'] = aa_bias['model'].apply(lambda x: plot_order.index(x))
    aa_groups = {'G': 'G',
    'A': 'aliphatic',
    'V': 'aliphatic',
    'L': 'aliphatic',
    'I': 'aliphatic',
    'P': 'P',
    'M': 'aliphatic',
    'F': 'aromatic',
    'Y': 'aromatic',
    'W': 'aromatic',
    'S': 'polar',
    'T': 'polar',
    'C': 'C',
    'N': 'polar',
    'Q': 'polar',
    'D': 'charged: acid',#'acid',
    'E': 'charged: acid',# 'acid',
    'R': 'charged: base',#'base',
    'H': 'H',
    'K': 'charged: base'}
    aa_bias['group'] = aa_bias['AA'].apply(lambda x: aa_groups[x])
    aa_bias = aa_bias.sort_values(by=['order','group']).reset_index(drop=True)

    fig,ax = plt.subplots(1,1,figsize=(7,5))
    sns.barplot(data=aa_bias,x='AA',y='frequency',hue='model',errorbar='sd',ax=ax)
    ax.set_ylabel('Frequency',fontsize=14)
    ax.set_xlabel('Mutant Amino Acid',fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
    ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks()],fontsize=12)
    ax.yaxis.grid()
    ax.legend(fontsize=13,bbox_to_anchor=(.78,.5))
    ax.set_title('Modelwise Sampling Bias in Top 10% of Predictions',fontsize=14)
    plt.savefig(FILE_PATH+'outputs/Fig3D.png',bbox_inches='tight')

def figure4():
    from utils import fold2ddG, fitAbRFC, applyAbRFC, fitAbRFR, applyAbRFR
    params = json.load(open(FILE_PATH+'code/params.json'))

    OUTPUT_DIR = FILE_PATH+'outputs/'
    X = pd.read_csv(FILE_PATH+'data/train/Xtrain.csv')
    DDG_CUTOFF = fold2ddG(params['fold_cutoff'])
    X['ID'] = X['ID'].astype(str)
    X.set_index('ID',inplace=True)
    X['aif_score'] =np.where(X['aif_score']=='False',np.NaN,X['aif_score'])
    X['y_clf'] = np.where(X['ddG']>=DDG_CUTOFF,1,0)
    CLF_PARAMS  = params['rf_params']
    CLF_PARAMS['random_state'] = 12068 #for reproducibility
    
    
    Xval = pd.read_csv(FILE_PATH+'data/validation/Xval.csv')
    clf = fitAbRFC(X.loc[:,params['features']].values,X.loc[:,'y_clf'].values,CLF_PARAMS)
    REG_PARAMS = copy.deepcopy(CLF_PARAMS)
    del REG_PARAMS['class_weight']
    del REG_PARAMS['criterion']
    reg = fitAbRFR(X.loc[:,params['features']].values,X.loc[:,'ddG'].values,REG_PARAMS)

    Xval['AbRFC'] = applyAbRFC(Xval.loc[:,params['features']].values,clf)
    Xval['RF Regressor'] = applyAbRFR(Xval.loc[:,params['features']].values,reg)
    #Xval = pd.merge(Xval.drop('GNN Classifier',axis=1),tmp[['ESM LM','GNN Classifier']],on='ESM LM')

    scores = Xval[['label', 'AbRFC','RF Regressor', 'GNN Classifier','GNN Regressor','GEO PPI','ESM LM','AbLang','relative_od']]
    scores['yval'] = (scores['relative_od']>=1).astype(int)


    ps = np.arange(.1,1.1,.1)
    rows = []
    oracle = []

    for p in ps:
        for model in ['Oracle', 'AbRFC','RF Regressor', 'GNN Classifier','GNN Regressor','GEO PPI','ESM LM','AbLang']:
            if model=='GNN Regressor':
                #the authors (and checking on the training set) indicate that lower DDG is better
                #all other models followed the GEO PPI convention of REF - MUT where higher DDG is better
                ix_consider = scores.sort_values(model,ascending=True).index.values[:int(len(scores)*p)]
            elif model=='Oracle':
                #in this case the best possible is sorting by the true relative od
                ix_consider = scores.sort_values('relative_od',ascending=False).index.values[:int(len(scores)*p)]

            else:
                ix_consider = scores.sort_values(model,ascending=False).index.values[:int(len(scores)*p)]
            mean_relative_od = scores.loc[ix_consider,'relative_od'].mean()
            rows+=[{'p':p,'model':model,'mean_relative_od':mean_relative_od}]

    benchmark_validation  = pd.DataFrame.from_records(rows)

    benchmark_validation['sortix'] = 10
    benchmark_validation['sortix'] = benchmark_validation['model'].replace({'AbRFC':0,'RF Regressor':1,
    'GNN Classifier':2,'AbLang':3,
    'Training Set Frequency':4})
    benchmark_validation = benchmark_validation.sort_values(['p','sortix']).reset_index(drop=True)


    ptree, rtree, ttree = precision_recall_curve(scores['yval'].values,scores['AbRFC'].values)
    preg, rreg, treg = precision_recall_curve(scores['yval'].values,scores['RF Regressor'].values)
    pablang, rablang, tablang = precision_recall_curve(scores['yval'].values,scores['AbLang'].values)
    pgnn, rgnn, tgnn = precision_recall_curve(scores['yval'].values,scores['GNN Classifier'].values)
    pgnnreg, rgnnreg, tgnnreg = precision_recall_curve(scores['yval'].values,scores['GNN Regressor'].values)
    pgppi, rgppi, tgppi = precision_recall_curve(scores['yval'].values,scores['GEO PPI'].values)
    pesm, resm, tesm = precision_recall_curve(scores['yval'].values,scores['ESM LM'].values)

    fig,ax = plt.subplots(2,1,figsize=(7,6))
    axa = ax[0]
    axa.plot(rtree,ptree,label='AbRFC (AUC={0:.2f})'.format(auc(rtree,ptree)))
    axa.plot(rreg,preg,label='RF Regressor (AUC={0:.2f})'.format(auc(rreg,preg)))
    axa.plot(rgnn,pgnn,label='GNN Classifier (AUC={0:.2f})'.format(auc(rgnn,pgnn)))
    axa.plot(rablang,pablang,label='AbLang (AUC={0:.2f})'.format(auc(rablang,pablang)))
    axa.plot(resm,pesm,label='ESM LM (AUC={0:.2f})'.format(auc(resm,pesm)))
    axa.plot(rgppi,pgppi,label='GEO PPI (AUC={0:.2f})'.format(auc(rgppi,pgppi)))
    axa.plot(rgnnreg,pgnnreg,label='GNN Regressor (AUC={0:.2f})'.format(auc(rgnnreg,pgnnreg)))
    axa.legend(fontsize=8,loc='upper left', bbox_to_anchor=(.85,1.1))

    random_precision    = scores['yval'].sum()/len(scores)
    axa.plot([0,1],[random_precision,random_precision],color='black',linestyle='--',label='Random')
    axa.set_xlabel('Recall',fontsize=10)
    axa.set_ylabel('Precision',fontsize=10)
    axa.set_title('Predicting Non-Deleterious Mutations',fontsize=10)
    axe = ax[1]
    sns.barplot(x='p',y='mean_relative_od',hue='model',data=benchmark_validation[benchmark_validation['model']!='Oracle'],ax=axe)
    axe.set_xlabel('Top N (Fraction) Mutations Selected',fontsize=10)
    axe.set_ylabel('Mean Binding Improvement\n(Relative OD)',fontsize=10)
    axe.set_xticklabels(['{}(.1)'.format(int(79*ps[0])),'{}(.2)'.format(int(79*ps[1])),
                        '{}(.3)'.format(int(79*ps[2])),'{}(.4)'.format(int(79*ps[3])),
                        '{}(.5)'.format(int(79*ps[4])),'{}(.6)'.format(int(79*ps[5])),
                        '{}(.7)'.format(int(79*ps[6])),'{}(.8)'.format(int(79*ps[7])),
                        '{}(.9)'.format(int(79*ps[8])),'{}(1)'.format(int(79*ps[9]))],fontsize=10)
    xmin = axe.get_xlim()[0]
    xmax = axe.get_xlim()[1]
    oracle = benchmark_validation[benchmark_validation['model']=='Oracle']
    axe.plot(axe.get_xticks(),oracle['mean_relative_od'].values,'k-.',label='Oracle')
    axe.plot([axe.get_xlim()[0],axe.get_xlim()[1]],[1,1],'k-',label='Non-Deleterious')
    axe.plot([axe.get_xlim()[0],axe.get_xlim()[1]],[0.79486,0.79486],'k--',label='Random')
    axe.set_xlim([xmin,xmax])
    axe.set_title('Enrichment of Affinity Enhancing Mutations',fontsize=10)
    axe.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1.1))
    axa.yaxis.grid()
    axe.yaxis.grid()
    plt.suptitle('OOD Validation Set Performance',fontsize=12)
    plt.tight_layout()
    plt.savefig(FILE_PATH+'outputs/Fig4.png',bbox_inches='tight')

def scoreCMAB():
    from utils import generateScores
    DATA_DIR = FILE_PATH+'/data'
    params = {'cdrs':{'H1':((26,34),),  ##SCORING RULES##
    'H2':((50,59),), ##SCORING RULES##
    'H3':((93,102),),##SCORING RULES##
    'L1':((27,34),), ##SCORING RULES##
    'L2':((50,56),), ##SCORING RULES##
    'L3':((89,99),)},##SCORING RULES##
    'per_position':7,##SCORING RULES##
    'refAA_max':95,##SCORING RULES##
    'mutAA_min':.01,##SCORING RULES##
    'n_interface':50,##SCORING RULES##
    'n_noninterface':25,##SCORING RULES##
    'pdb':'{}/test/CMAB0/7LOP_omicron.pdb'.format(DATA_DIR), ##SCORING DATA##
    'feature_files':{'6YLA':'{}/test/CMAB0/6YLA_scores.pickle'.format(DATA_DIR),##SCORING DATA##
                     '7LOP':'{}/test/CMAB0/7LOP_scores.pickle'.format(DATA_DIR)},##SCORING DATA##
    'random_state':6777973}
    scores = generateScores(params)
    scores.to_csv(FILE_PATH+'outputs/CMAB_scores.csv',index=False)
    return scores

def scoreGMAB():
    from utils import generateScores
    DATA_DIR = FILE_PATH+'/data'
    params = {'cdrs':{'H1':((26,34),),  ##SCORING RULES##
    'H2':((50,59),), ##SCORING RULES##
    'H3':((93,102),),##SCORING RULES##
    'L1':((27,34),), ##SCORING RULES##
    'L2':((50,56),), ##SCORING RULES##
    'L3':((89,99),)},##SCORING RULES##
    'per_position':4,##SCORING RULES##
    'refAA_max':95,##SCORING RULES##
    'mutAA_min':.01,##SCORING RULES##
    'n_interface':31,##SCORING RULES##
    'n_noninterface':19,##SCORING RULES##
    'pdb':'{}/test/GMAB0/7BEP_omicron.pdb'.format(DATA_DIR), ##SCORING DATA##
    'feature_files':{'7BEP':'{}/test/GMAB0/7BEP_scores.pickle'.format(DATA_DIR),##SCORING DATA##
                     '7R6W':'{}/test/GMAB0/7R6W_scores.pickle'.format(DATA_DIR)},##SCORING DATA##
    'random_state':3157581, #reproducibility
    }
    scores = generateScores(params)
    scores.to_csv(FILE_PATH+'outputs/GMAB_scores.csv',index=False)
    return scores

def figure6(scores_cmab, scores_gmab):
    import matplotlib
    from utils import numstr2tup
    def n2t(_x,i):
        try: 
            return numstr2tup(_x.split('_')[0][1:])[i]
        except:
            return 0

    def l2c(_x):
        if _x[0] in ['H','L']:
            return _x[0]
        else:
            return 'C'

    data_cmab   = pd.read_excel(FILE_PATH+'data/test/CMAB0/CMAB_ELISA_OD.xlsx') 
    data_cmab = data_cmab[~data_cmab['Processed OD (.06 ug/mL)'].isna()].copy().reset_index(drop=True)
    data_cmab.rename(columns={'Processed OD (.06 ug/mL)':'OMRBD_OD'},inplace=True)

    data_gmab   = pd.read_excel(FILE_PATH+'data/test/GMAB0/GMAB_ELISA_OD.xlsx')
    data_gmab.rename(columns={'Processed OD (.3 ug/mL)':'OMRBD_OD'},inplace=True)

    selected_gmab = list(scores_gmab[scores_gmab.model_selected==1].label.values)+\
                    ['GMAB0','H54_NT'] #H54_NT is the deamination mutation
    selected_cmab = list(scores_cmab[scores_cmab.model_selected==1].label.values)+\
                    ['CMAB0'] 
    data_gmab = data_gmab[data_gmab['label'].isin(selected_gmab)].reset_index(drop=True)
    data_gmab['label'] = data_gmab['label'].replace({'GMAB0':'S309'})
    data_gmab['chain'] = data_gmab['label'].apply(lambda x: l2c(x))
    data_gmab['resn'] = data_gmab['label'].apply(lambda x: n2t(x,0))
    data_gmab['resi'] = data_gmab['label'].apply(lambda x: n2t(x,1))
    data_gmab = data_gmab.sort_values(['chain','resn','resi']).reset_index(drop=True)

    data_cmab = data_cmab[data_cmab['label'].isin(selected_cmab)].reset_index(drop=True)
    data_cmab['label'] = data_cmab['label'].replace({'GMAB0':'S309'})
    data_cmab['chain'] = data_cmab['label'].apply(lambda x: l2c(x))
    data_cmab['resn'] = data_cmab['label'].apply(lambda x: n2t(x,0))
    data_cmab['resi'] = data_cmab['label'].apply(lambda x: n2t(x,1))
    data_cmab = data_cmab.sort_values(['chain','resn','resi']).reset_index(drop=True)


    fig,ax = plt.subplots(3,2,figsize=(16,14))

    ## GMAB VH
    sns.barplot(data = data_gmab[data_gmab.chain.isin(['H','C'])],x='label', y='OMRBD_OD',ax=ax[0,0],palette=sns.dark_palette('Blue',n_colors=50))
    ax[0,0].set_xticklabels([x.get_text().replace('_',':') for x in ax[0,0].get_xticklabels()],rotation=90,fontsize=14,va='bottom')
    gmab156 = 'H31_SK,H54_NT,H100_AS'.replace('_',':').split(',')
    dx = 0/72.; dy = -68/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for i,label in enumerate(ax[0,0].xaxis.get_majorticklabels()):
        label.set_transform(label.get_transform() + offset)
        #print(label.get_unitless_position())
        if label.get_text()=='H54:NT':
            ax[0,0].text(i,-.57,'*',fontsize=14,ha='center')
        if label.get_text() in gmab156:
            ax[0,0].text(i,-.54,'^',fontsize=18,ha='center')

    ax[0,0].set_ylabel(r'OD (.3 $\mu \mathrm{g}$/mL)',fontsize=14)
    gmab_mean = data_gmab.loc[data_gmab.chain.isin(['C']),'OMRBD_OD'].mean()
    ax[0,0].axhline(data_gmab.loc[data_gmab.chain.isin(['C']),'OMRBD_OD'].mean(),c='k')
    ax[0,0].set_xlabel('')
    ax[0,0].set_title('GMAB Series VH Mutation Screening\n Omicron BA.1 RBD',fontsize=16)
    ##

    ## GMAB VL
    sns.barplot(data = data_gmab[data_gmab.chain.isin(['L','C'])],x='label', y='OMRBD_OD',ax=ax[0,1],palette=sns.dark_palette('Blue',n_colors=50))
    ax[0,1].set_xticklabels([x.get_text().replace('_',':') for x in ax[0,1].get_xticklabels()],rotation=90,fontsize=14,va='bottom')
    gmab156 = 'L28_TE,L52_SY,L93_TE'.replace('_',':').split(',')
    dx = 0/72.; dy = -55/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for i,label in enumerate(ax[0,1].xaxis.get_majorticklabels()):
        label.set_transform(label.get_transform() + offset)
        #print(label.get_unitless_position())
        if label.get_text() in gmab156:
            ax[0,1].text(i,-.43,'^',fontsize=18,ha='center')

    ax[0,1].set_ylabel(r'OD (.3 $\mu \mathrm{g}$/mL)',fontsize=14)
    ax[0,1].axhline(data_gmab.loc[data_gmab.chain.isin(['C']),'OMRBD_OD'].mean(),c='k')
    ax[0,1].set_xlabel('')
    ax[0,1].set_title('GMAB Series VL Mutation Screening\n Omicron BA.1 RBD',fontsize=16)

    ## CMAB VH
    # #C CMAB0 VH PMs
    sns.barplot(data = data_cmab[data_cmab.chain.isin(['H','C'])],x='label', y='OMRBD_OD',ax=ax[1,0],palette=sns.dark_palette('Blue',n_colors=50))
    ax[1,0].set_xticklabels([x.get_text().replace('_',':') for x in ax[1,0].get_xticklabels()],rotation=90,fontsize=14,va='bottom')

    cmab262 = 'H55_SQ,H99_SW'.replace('_',':').split(',')
    dx = 0/72.; dy = -60/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for i,label in enumerate(ax[1,0].xaxis.get_majorticklabels()):
        label.set_transform(label.get_transform() + offset)
        #print(label.get_unitless_position())
        if label.get_text() in cmab262:
            ax[1,0].text(i,-.34,'^',fontsize=18,ha='center')

    ax[1,0].set_ylabel(r'OD (.06 $\mu \mathrm{g}$/mL)',fontsize=14)
    ax[1,0].axhline(data_cmab.loc[data_cmab.chain.isin(['C']),'OMRBD_OD'].mean(),c='k')
    ax[1,0].set_xlabel('')
    ax[1,0].set_title('CMAB Series VH Mutation Screening\nOmicron BA.1 RBD',fontsize=16)
    ##

    ## CMAB VL
    sns.barplot(data = data_cmab[data_cmab.chain.isin(['L','C'])],x='label', y='OMRBD_OD',ax=ax[1,1],palette=sns.dark_palette('Blue',n_colors=50))
    ax[1,1].set_xticklabels([x.get_text().replace('_',':') for x in ax[1,1].get_xticklabels()],rotation=90,fontsize=14,va='bottom')

    cmab262 = 'L31_NE,L56_SY,L90_QN,L93_ST'.replace('_',':').split(',')
    dx = 0/72.; dy = -60/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for i,label in enumerate(ax[1,1].xaxis.get_majorticklabels()):
        label.set_transform(label.get_transform() + offset)
        #print(label.get_unitless_position())
        if label.get_text() in cmab262:
            ax[1,1].text(i,-.3,'^',fontsize=18,ha='center')

    ax[1,1].set_ylabel(r'OD (.06 $\mu \mathrm{g}$/mL)',fontsize=14)
    cmab_mean = data_cmab.loc[data_cmab.chain.isin(['C']),'OMRBD_OD'].mean()
    ax[1,1].axhline(data_cmab.loc[data_cmab.chain.isin(['C']),'OMRBD_OD'].mean(),c='k')
    ax[1,1].set_xlabel('')
    ax[1,1].set_title('CMAB Series VL Mutation Screening\nOmicron BA.1 RBD',fontsize=16)


    # #ax[2,0].set_title('Relative ELISA EC50s of Round 2 GMAB Series Designs',fontsize=16,va='top')
    # #ax[2,1].set_title('Relative ELISA EC50s of Round 2 CMAB Series Designs',fontsize=16,va='top')
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,0].axis('off')
    ax[2,1].set_xticks([])
    ax[2,1].set_yticks([])
    ax[2,1].axis('off')
    plt.tight_layout()
    plt.savefig(FILE_PATH+'outputs/Fig6.png',bbox_inches='tight')


    


if __name__ == '__main__':
    from utils import run_cv
    run_cv()
    figures3CS1()
    figure3A()
    figure3B()
    figure3D()
    figure4()
    scores_cmab = scoreCMAB()
    scores_gmab = scoreGMAB()
    figure6(scores_cmab,scores_gmab)

