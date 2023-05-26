from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import json
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import copy


def fold2ddG(fold,T=298):
    """takes a fold value (Kd_wt/Kd_mut) and make it a fold change
       ddG is in Kcal/mol
       R is 1.9872 cal/(K*mol)
       T is in kelvin
       """
    R = 1.9872/1000 #kcal
    return R*T*np.log(fold)

def feature_importances(clf,features_keep,fold):
     ### Calculate the feature importances given a random forest classifier
     ##  values for all trees in the forest are used as to estimate the distribution.
     ##  graph is ranked based on median and may fluctuate slightly from run to run.
     fimps = []
     rf = clf.steps[1][1]
     std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
     for k in range(len(rf.feature_importances_)):
          fimps+=[{'feature':features_keep[k],'importance':np.round(rf.feature_importances_[k],3),
                              'importance_std':np.round(std[k],3), 'fold':fold}]
     return fimps

def plot_mutation_frequencies(mtrain,mval):
    ## this code generates the mutation frequency heatmap in Figure 2A and 2B
    ## Note that the mutation frequencies are precomputed, but replicating the 
    ## training data frequency heatmap is straightforward using the data in 
    ## X.csv.  Note that the data X contains both forward and reverse mutations
    ## for SKEMPI, but only the forward mutations (ID<906) are used for figure 2A.

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    axa, axb = ax
    sns.heatmap(mtrain,annot=True,cmap='Blues',
                linewidths=1,linecolor='black',fmt='g',cbar=False,ax=axa)
    axa.set_xlabel('Mutated AA',fontsize=14)
    axa.set_ylabel('Reference AA',fontsize=14)
    axa.set_title('Training Set Mutation Frequencies',fontsize=14)
    sns.heatmap(mval,annot=True,cmap='Blues',
                linewidths=1,linecolor='black',fmt='g',cbar=False,ax=axb)
    axb.set_xlabel('Mutated AA',fontsize=14)
    axb.set_ylabel('Reference AA',fontsize=14)
    axb.set_title('Validation Set Mutation Frequencies',fontsize=14)
    plt.tight_layout()
    plt.savefig('./outputs/Figure2AB.pdf',dpi=300)


def model_bias(scores_aa,p=.25):
    ###replicates figure 2C from the paper
    ###a lot of code, but it simply counts up the number of times an
    ###AA is scored in the top 25% of predictions for each model
    ###note that the code was written at first trying to assume a random model,
    ###but quickly realized that the random model was equivalent to training set
    ###frequency so that's why the Training Set Frequency is calulated as it is.
    bias_counts = {'AbRFC':{y:0 for y in 'ACDEFGHIKLMNPQRSTVWY'},
                'ABL':{y:0 for y in 'ACDEFGHIKLMNPQRSTVWY'},
                'GNN Classifier':{y:0 for y in 'ACDEFGHIKLMNPQRSTVWY'},
                'RF Regressor':{y:0 for y in 'ACDEFGHIKLMNPQRSTVWY'},
                'Random':{y:0 for y in 'ACDEFGHIKLMNPQRSTVWY'}}
    # dup = {}
    # for k, v in scores_aa.items():
    #     dup[int(k)] = v
    # scores_aa = dup
    # del dup

    N_RANDOM = 0
    N_MODELS = 0
    for FOLD in range(5):
        N = int(len(scores_aa[FOLD]['YTEST'])*p)
        for MODEL in ['AbRFC','RF Regressor', 'GNN Classifier','ABL','Random']:
            if MODEL in ['AbRFC','RF Regressor', 'GNN Classifier','ABL']:
                if MODEL=='ABL':
                    N_MODELS+=N
                ix_model = np.argsort(scores_aa[FOLD][MODEL])[::-1][:N]
            else:
                ix_model = np.random.choice(len(scores_aa[FOLD]['YTEST']),len(scores_aa[FOLD]['YTEST']),replace=False)
                N_RANDOM+=len(ix_model)
            WT_AA = np.array(scores_aa[FOLD]['WT_AA'])[ix_model]
            MT_AA = np.array(scores_aa[FOLD]['MT_AA'])[ix_model]
            for i in range(len(WT_AA)):
                bias_counts[MODEL][MT_AA[i]]+=1

    to_plot = []
    model_names = {'AbRFC':'AbRFC','RF Regressor':'RF Regressor','GNN Classifier':'GNN Classifier','ABL':'AbLang','Random':'Training Set Frequency'}
    for k, v in bias_counts.items():
        for kk, vv in v.items():
            if k!='Random':
                to_plot.append({'model':model_names[k],'Mutated AA':kk,'count':vv})
            else:
                to_plot.append({'model':model_names[k],'Mutated AA':kk,'count':vv*N_MODELS/N_RANDOM})


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
    'D': 'charged: acid',
    'E': 'charged: acid',
    'R': 'charged: base',
    'H': 'H',
    'K': 'charged: base'}

    to_plot = pd.DataFrame.from_records(to_plot)
    to_plot['aa_group'] = to_plot['Mutated AA'].apply(lambda x: aa_groups[x])

    to_plot['sortix'] = to_plot['model'].replace({'AbRFC':0,'RF Regressor':1,
    'GNN Classifier':2,'AbLang':3,
    'Training Set Frequency':4})
    to_plot = to_plot.sort_values(['aa_group','sortix']).reset_index(drop=True)

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    sns.barplot(x='Mutated AA',y='count',hue='model',data=to_plot[to_plot['Mutated AA'].isin(['A','C'])==False],ax=ax)
    ax.yaxis.grid()
    plt.savefig('./outputs/Figure2C.pdf',dpi=300)

def crossval(features_keep,CLF_PARAMS,X,CV,GNNPATH,ABLANGFILE):
    fig, ax = plt.subplots(3,2,figsize=(7,7))
    spearmanrs = []
    all_fimps = []

    scores_aa = {}
    REG_PARAMS   = copy.deepcopy(CLF_PARAMS['rf_params'])
    del REG_PARAMS['class_weight']
    del REG_PARAMS['criterion']

    ##generates figure S1
    for FOLD in range(5):
        pltx = FOLD%2
        plty = FOLD//2
        ix_train, ix_test = CV[str(FOLD)]

        
        abl = pd.read_csv(ABLANGFILE)
        abl['ablang_preds'] = abl['abl_pm']
        ix_train_abl = list(set(ix_train).intersection(set(abl.ID.unique())))
        ix_test_abl =  list(set(ix_test).intersection(set(abl.ID.unique())))
        ypred_abl = abl.set_index('ID').loc[ix_test_abl,'ablang_preds'].values
        
        ytest = X.loc[ix_test,'y_clf'].values
        ytest_abl = X.loc[ix_test_abl,'y_clf'].values
        ytrain = X.loc[ix_train,'y_clf'].values
        ytrain_reg  = X.loc[ix_train,'ddG'].values

        #####TREE######
        xtrain_tree = X.loc[ix_train,features_keep].values
        imp = IterativeImputer(max_iter=10)
        xtrain_tree = imp.fit_transform(xtrain_tree)
        clf = make_pipeline(StandardScaler(),RandomForestClassifier(**CLF_PARAMS['rf_params']))
        clf.fit(xtrain_tree,ytrain)

        all_fimps+=feature_importances(clf,features_keep,FOLD)

        #tree preds
        ypred = clf.predict_proba(imp.transform(X.loc[ix_test,features_keep].values))[:,1]
        ypred_abl_subset = clf.predict_proba(imp.transform(X.loc[ix_test_abl,features_keep].values))[:,1]
        ####TREE END####

        #####REGRESSION####
        xtrain_reg = X.loc[ix_train,features_keep].values
        imp = IterativeImputer(max_iter=10)
        xtrain_reg = imp.fit_transform(xtrain_reg)
        reg = make_pipeline(StandardScaler(),RandomForestRegressor(**REG_PARAMS))
        reg.fit(xtrain_reg,ytrain_reg)
        ypred_reg = reg.predict(imp.transform(X.loc[ix_test,features_keep].values))
        ypred_reg_abl_subset = reg.predict(imp.transform(X.loc[ix_test_abl,features_keep].values))
        ####REGRESSION END####

        ####GNN####
        gnn = pd.read_csv(GNNPATH+'cv_fold_{}_preds.csv'.format(FOLD))
        gnn['ID'] = gnn['ID'].astype(str)

        #match means all ix_train in gnn_train AND no gnn_train in ix_test
        gnn_train = gnn.loc[gnn['is_train']==1,'ID'].values
        train_match = np.all(np.in1d(ix_train,gnn_train))
        val_match = ~np.any(np.in1d(gnn_train,ix_test))

        #train_match = np.all(gnn.loc[gnn['is_train']==1,'ID'].apply(lambda x: x in ix_train)) and np.all([x in gnn.loc[gnn['is_train']==1,'ID'].values for x in ix_train])
        #val_match =  np.all(gnn.loc[gnn['is_train']==0,'ID'].apply(lambda x: x in ix_test)) and np.all([x in gnn.loc[gnn['is_train']==0,'ID'].values for x in ix_test])
        assert train_match and val_match
        #ypred_gnn_train = gnn.set_index('ID').loc[ix_train,'model_preds'].values
        ypred_gnn = gnn.set_index('ID').loc[ix_test,'model_preds'].values
        ypred_gnn_abl = gnn.set_index('ID').loc[ix_test_abl,'model_preds'].values
        ####GNN END####
        
        ####PR CURVES####
        ptree, rtree, ttree = precision_recall_curve(ytest,ypred)
        preg, rreg, treg = precision_recall_curve(ytest,ypred_reg)
        pgnn, rgnn, tgnn = precision_recall_curve(ytest,ypred_gnn)#ypred_gnn)
        pabl, rabl, tabl = precision_recall_curve(ytest_abl,ypred_abl)
        #prandom, rrandom, trandom =  precision_recall_curve(np.random.permutation(ytest),ypred)
        random_precision = ytest.sum()/len(ytest)
        print('Fold {0}: Random Precision={1:.2f}'.format(FOLD,random_precision))
        ax[plty,pltx].plot(rtree,ptree,label='TREE')
        ax[plty,pltx].plot(rreg,preg,label='RF REGRESSION')
        ax[plty,pltx].plot(rgnn,pgnn,label=' GNN')
        ax[plty,pltx].plot(rabl,pabl,label='ABLANG')
        #ax[plty,pltx].plot(rrandom,prandom,label='RANDOM')

        ax[plty,pltx].plot([0,1],[random_precision,random_precision],'k--',label='Baseline')
        ax[plty,pltx].set_ylim(.5,1.05)
        ax[plty,pltx].set_ylim(.5,1.05)
        ax[plty,pltx].set_title('Fold %s '%FOLD,fontsize=12)
        ax[plty,pltx].set_xlabel('Recall',fontsize=12)
        ax[plty,pltx].set_ylabel('Precision',fontsize=12)

        spearmanrs+= [{'Fold':FOLD, 'model_1':'TREE','model_2':'GNN','spearman':spearmanr(ypred,ypred_gnn)[0].round(2)},
                {'Fold':FOLD, 'model_1':'TREE','model_2':'ABLANG','spearman':spearmanr(ypred_abl_subset,ypred_abl)[0].round(2)},
                {'Fold':FOLD, 'model_1':'GNN','model_2':'ABLANG','spearman':spearmanr(ypred_gnn_abl,ypred_abl)[0].round(2)}]
        
        ax[plty,pltx].text(.1,.52, '% Ala train: {0:.1f} test: {1:.1f}'.format( np.sum((X.loc[ix_train,'mutAA']=='A')|(X.loc[ix_train,'refAA']=='A'))/len(ix_train)*100,
                                                                            np.sum((X.loc[ix_test, 'mutAA']=='A')|(X.loc[ix_test, 'refAA']=='A'))/len(ix_test)*100))
        
        ax[plty,pltx].grid()
        mut_aas = X.loc[ix_test,'mutAA'].values
        ix_consider = np.where(mut_aas!='A')[0]
        aas_consider = mut_aas[ix_consider]
        
        ytest_consider = ytest[ix_consider]
        ypred_consider = ypred[ix_consider]
        ypred_gnn_consider = ypred_gnn[ix_consider]
        ref_aas_abl = X.loc[ix_test_abl,'refAA'].values
        mut_aas_abl = X.loc[ix_test_abl,'mutAA'].values
        ix_consider_abl = np.where(mut_aas_abl!='A')[0]

        scores_aa[FOLD] = {'AbRFC':list(ypred_abl_subset[ix_consider_abl]),
                        'RF Regressor':list(ypred_reg_abl_subset[ix_consider_abl]),
                            'GNN Classifier':list(ypred_gnn_abl[ix_consider_abl]),
                            'YTEST':list(ytest_abl[ix_consider_abl]),
                            'MT_AA':list(mut_aas_abl[ix_consider_abl]),
                            'WT_AA':list(ref_aas_abl[ix_consider_abl]),
                            'ABL':list(ypred_abl[ix_consider_abl]),
                            }
        
    ax[2,1].plot([.5,.5],[.5,.5],label='AbRFC')
    ax[2,1].plot([.5,.5],[.5,.5],label='RF Regressor')
    ax[2,1].plot([.5,.5],[.5,.5],label='GNN Classifier')
    ax[2,1].plot([.5,.5],[.5,.5],label='AbLang')
    ax[2,1].plot([.5,.5],[.5,.5],'k--',label='Baseline')
    ax[2,1].legend(bbox_to_anchor=[.5,.5],loc='center',fontsize=14)
    ax[2,1].set_xticks([])
    ax[2,1].set_yticks([])
    ax[2,1].axis('off')
    plt.suptitle('Cross Validation Precision Recall Curves',fontsize=12)
    plt.tight_layout()
    plt.savefig('./outputs/FigureS1.pdf',dpi=300)
    plt.close()
    #generate figure 2D
    fig2d = pd.DataFrame.from_records(all_fimps)
    fig2d = pd.merge(fig2d,fig2d.groupby('feature')['importance'].median().reset_index().rename(columns={'importance':'median_importance'}))
    fig2d = fig2d.sort_values('median_importance',ascending=False).reset_index(drop=True)
    fig, axd = plt.subplots(1,1,figsize=(10,5))
    sns.boxplot(x='feature',y='importance',data=fig2d,ax=axd)
    axd.set_xticklabels(axd.get_xticklabels(),rotation=50,ha='right',rotation_mode='anchor')
    axd.yaxis.grid()
    axd.set_title('AbRFC Feature Importance\n During Cross Validation',fontsize=14)
    axd.set_ylabel('Mean Decrease in\n Gini Impurity',fontsize=14)
    axd.set_xlabel('')
    plt.tight_layout()
    plt.savefig('./outputs/Figure2D.pdf',dpi=300)

    #generate model bias plot (Figure 2C)
    model_bias(scores_aa)
    
def benchmark_validation(X,CLF_PARAMS,features_keep,cutoff=.7):
    ###Code to reproduce the benchmark validation results
    ###as ABRFC and the RF Regressor are the novel models of this work,
    ###they are trained and tested here for reproducibility.  All other scores
    ###were precomputed using open source code / models and are in the data folder.
    ###PARAMS: `cutoff`` is the fold change cutoff used to convert SKEMPI ddG values to binary
    ###        labels for training AbRFC.

    REG_PARAMS   = copy.deepcopy(CLF_PARAMS['rf_params'])
    del REG_PARAMS['class_weight']
    del REG_PARAMS['criterion']
    ix_train = X.index[X.dataset=='SKEMPI']
    ix_test  = X.index[X.dataset=='VALIDATION']
    cutoff = fold2ddG(cutoff)
    xtrain_tree = X.loc[ix_train,features_keep].values
    imp = IterativeImputer(max_iter=10)
    xtrain_tree = imp.fit_transform(xtrain_tree)
    ytrain = (X.loc[ix_train,'ddG'].values>=cutoff).astype(int)
    clf = make_pipeline(StandardScaler(),RandomForestClassifier(**CLF_PARAMS['rf_params']))
    reg = make_pipeline(StandardScaler(),RandomForestRegressor(**REG_PARAMS))
    clf.fit(xtrain_tree,ytrain)
    reg.fit(xtrain_tree,X.loc[ix_train,'ddG'].values)
    ytest = X.loc[ix_test,'relative_od'].values

    ypred = clf.predict_proba(X.loc[ix_test,features_keep].values)[:,1]
    ypred_reg = reg.predict(X.loc[ix_test,features_keep].values)

    scores = pd.DataFrame({'ID':X.loc[ix_test,'ID'].values,'AbRFC':ypred,'RF Regressor':ypred_reg,'relative_od':ytest})
    scores_other = pd.read_csv('./data/validation/benchmark_validation_scores.csv')
    scores['ID'] = scores['ID'].astype(str)
    scores_other['ID'] = scores_other['ID'].astype(str)
    scores = pd.merge(scores,scores_other)
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

    bmv  = pd.DataFrame.from_records(rows)

    bmv['sortix'] = 10
    bmv['sortix'] = bmv['model'].replace({'AbRFC':0,'RF Regressor':1,
    'GNN Classifier':2,'AbLang':3,
    'Training Set Frequency':4})
    bmv = bmv.sort_values(['p','sortix']).reset_index(drop=True)



    fig,axe = plt.subplots(1,1,figsize=(12,4))
    sns.barplot(x='p',y='mean_relative_od',hue='model',data=bmv[bmv['model']!='Oracle'],ax=axe)
    axe.set_xlabel('Top N (%) Mutations Selected',fontsize=14)
    axe.set_ylabel('Mean Binding Improvement\n(relative OD)',fontsize=14)
    axe.set_xticklabels(['{} (10)'.format(int(87*ps[0])),'{} (20)'.format(int(87*ps[1])),
                        '{} (30)'.format(int(87*ps[2])),'{} (40)'.format(int(87*ps[3])),
                        '{} (50)'.format(int(87*ps[4])),'{} (60)'.format(int(87*ps[5])),
                        '{} (70)'.format(int(87*ps[6])),'{} (80)'.format(int(len(ytest)*ps[7])),
                        '{} (90)'.format(int(87*ps[8])),'{} (100)'.format(int(len(ytest)*ps[9]))])
    xmin = axe.get_xlim()[0]
    xmax = axe.get_xlim()[1]
    oracle = bmv[bmv['model']=='Oracle']
    axe.plot(axe.get_xticks(),oracle['mean_relative_od'].values,'k-.',label='Oracle')
    axe.plot([axe.get_xlim()[0],axe.get_xlim()[1]],[1,1],'k-',label='Non-Deleterious')
    axe.plot([axe.get_xlim()[0],axe.get_xlim()[1]],[0.79486,0.79486],'k--',label='Random')
    axe.set_xlim([xmin,xmax])
    axe.set_title('Validation Set Performance',fontsize=14)
    axe.legend(loc='center',bbox_to_anchor=(.66,.85),ncol=4)
    plt.savefig('./outputs/Figure2E.pdf',dpi=300)





     

def score_dataset(features_keep,CLF_PARAMS,X,datasetlist):
    scores_aa = {}
    ix_train = X[X['dataset']=='SKEMPI'].index
    ytrain = X.loc[ix_train,'y_clf'].values

    #####TREE######
    xtrain_tree = X.loc[ix_train,features_keep].values
    imp = IterativeImputer(max_iter=10)
    xtrain_tree = imp.fit_transform(xtrain_tree)
    clf = make_pipeline(StandardScaler(),RandomForestClassifier(**CLF_PARAMS['rf_params']))
    clf.fit(xtrain_tree,ytrain)

    for ScoreDataset in datasetlist:
          ix_test = X[X['dataset']==ScoreDataset].index
          #tree preds
          ypred = clf.predict_proba(imp.transform(X.loc[ix_test,features_keep].values))[:,1]
          ####TREE END####    
                         
          mut_aas = X.loc[ix_test,'mutAA'].values
          ix_consider = np.where(mut_aas!='A')[0]
          aas_consider = mut_aas[ix_consider]
          ypred_consider = ypred[ix_consider]


          scores_aa[ScoreDataset] = {'TREE':ypred_consider,
                                   'AA':aas_consider}

          scoredf = X[X['dataset']==ScoreDataset].copy()
          scoredf['TREE']= scores_aa[ScoreDataset]['TREE']
          scoredf['AA']= scores_aa[ScoreDataset]['AA']
          scoredf.reset_index()[['ID','AA','TREE']].to_csv('./outputs/scores_'+ScoreDataset+'.csv',index=False)
          
          