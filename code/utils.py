from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import json
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def feature_importances(clf,features_keep,fold):
     fimps = []
     rf = clf.steps[1][1]
     std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
     for k in range(len(rf.feature_importances_)):
          fimps+=[{'feature':features_keep[k],'importance':np.round(rf.feature_importances_[k],3),
                              'importance_std':np.round(std[k],3), 'fold':fold}]
     return fimps


def crossval(features_keep,CLF_PARAMS,X,CV,GNNPATH,ABLANGFILE):
    fig, ax = plt.subplots(3,2,figsize=(7,7))
    spearmanrs = []
    all_fimps = []

    scores_aa = {}

    for FOLD in range(5):
        pltx = FOLD%2
        plty = FOLD//2
        ix_train, ix_test = CV[str(FOLD)]

        
        abl = pd.read_csv(ABLANGFILE)
        abl['ablang_preds'] = abl['abl_pm']#)#np.min(0,np.log((abl['abl_pm']/abl['abl_pr'])))
        ix_train_abl = list(set(ix_train).intersection(set(abl.ID.unique())))
        ix_test_abl =  list(set(ix_test).intersection(set(abl.ID.unique())))
        ypred_abl = abl.set_index('ID').loc[ix_test_abl,'ablang_preds'].values
        
        ytest = X.loc[ix_test,'y_clf'].values
        ytest_abl = X.loc[ix_test_abl,'y_clf'].values
        ytrain = X.loc[ix_train,'y_clf'].values
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

        #feature importances

        ####TREE END####

        ####GNN####
        gnn = pd.read_csv(GNNPATH+'cv_fold_{}_preds.csv'.format(FOLD))
        gnn['ID'] = gnn['ID'].astype(str)
        gnn_train = gnn.loc[gnn['is_train']==1,'ID'].values
        gnn_test  = gnn.loc[gnn['is_train']==0,'ID'].values

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
        pgnn, rgnn, tgnn = precision_recall_curve(ytest,ypred_gnn)#ypred_gnn)
        pabl, rabl, tabl = precision_recall_curve(ytest_abl,ypred_abl)
        #prandom, rrandom, trandom =  precision_recall_curve(np.random.permutation(ytest),ypred)
        random_precision = ytest.sum()/len(ytest)
        print('Random Precision: %s'%random_precision)
        ax[plty,pltx].plot(rtree,ptree,label='TREE')
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

        mut_aas_abl = X.loc[ix_test_abl,'mutAA'].values
        ix_consider_abl = np.where(mut_aas_abl!='A')[0]

        scores_aa[FOLD] = {'TREE':ypred_abl_subset[ix_consider_abl],#ypred_consider,
                            'GNN':ypred_gnn_abl[ix_consider_abl],#ypred_gnn_consider,
                            'YTEST':ytest_abl[ix_consider_abl],#ytest_consider,
                            'AA':mut_aas_abl[ix_consider_abl],#aas_consider,
                            'ABL':ypred_abl[ix_consider_abl],
                            'ABL_YTEST':ytest_abl[ix_consider_abl],
                            'ABL_AA':mut_aas_abl[ix_consider_abl]}

    ax[2,1].plot([.5,.5],[.5,.5],label='TREE')
    ax[2,1].plot([.5,.5],[.5,.5],label='GNN')
    ax[2,1].plot([.5,.5],[.5,.5],label='ABLANG')
    ax[2,1].plot([.5,.5],[.5,.5],'k--',label='Baseline')
    ax[2,1].legend(bbox_to_anchor=[.5,.5],loc='center',fontsize=14)
    ax[2,1].set_xticks([])
    ax[2,1].set_yticks([])
    ax[2,1].axis('off')
    plt.suptitle('Cross Validation Precision Recall Curves:\nTREE and GNN Models',fontsize=12)
    plt.tight_layout()
   
    plt.savefig('FigureS1A.pdf',bbox_inches='tight')
    pd.DataFrame.from_records(spearmanrs).set_index(['Fold','model_1','model_2']).unstack('model_2').sort_index().to_excel('table_s7a.xlsx')
    all_fimps = pd.DataFrame.from_records(all_fimps)
    all_fimps = pd.merge(all_fimps,all_fimps.groupby('feature')['importance'].median().reset_index().rename(columns={'importance':'median_importance'}))

    fig, ax = plt.subplots(1,1,figsize=(7,2))
    sns.boxplot(data=all_fimps.sort_values('median_importance',ascending=False),x='feature',y='importance')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=10)
    ax.set_xlabel('Feature',fontsize=12)
    ax.set_ylabel('Mean Decrease in Impurity',fontsize=12)
    ax.yaxis.grid()
    plt.savefig('FigureS1B.pdf',bbox_inches='tight')

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
          scoredf.reset_index()[['ID','AA','TREE']].to_csv('scores_'+ScoreDataset+'.csv',index=False)
          
          