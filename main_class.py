import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,confusion_matrix
import sklearn.metrics as sm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 

# Format Graphics
plt.rc('text', usetex=True)
font = {'size'   : 22}
plt.rc('font', **font)

def combine_gics(write_flag=False):
    '''
    Utility function to combine and format Bloomberg 
    GICS data into single file.  In addition, we 
    develop a mapping between compustat files 
    and the GICS classifications 
    for each based on their cusips 
    '''

    df1 = pd.read_csv('gics1.csv')
    df2 = pd.read_csv('gics2.csv')
    df3 = pd.read_csv('gics3.csv')
    df4 = pd.read_csv('gics4.csv')

    dfs = [df1,df2,df3,df4]

    fdf = pd.concat([df1,df2,df3,df4])

    lastdf = fdf.dropna()
    lastdf = lastdf[lastdf.columns[1:]]
    lastdf.columns = ['cusip'] + list(lastdf.columns[1:])
    lastdf = lastdf.reset_index()

    lastdf['cusip'] = [_.split('/')[-1] for _ in lastdf.cusip]

    lastdf = lastdf[lastdf.columns[1:]]

    lastdf.iloc[:,1:5] = lastdf.iloc[:,1:5].astype(int)

    flepth = '/home/stevelinux/GDrive/Frank_Proj/data/src/allgics.csv'

    if write_flag:
        lastdf.to_csv(flepth)

    return lastdf 

def feature_build(df): 
    '''
    Add several financial ratios that are constructed from firm 
    fundamental data based upon definitions given in:  
    
    Financial Ratios: 
    https://www.jstor.org/stable/246645?seq=1#page_scan_tab_contents

    ccdc_f - Cash/Current Debt CASH/CD 
    cs_f - Cash/Sales CASH/S
    cta_f - Cash/Total Assets CASH/TA
    cd_f - Cash/Total Debt CASH/TD
    cfe_f - Cash Flow/Equity CFFO/EQ
    cfta_f - Cash Flow/Total Assets CFFO/TA
    cftd_f - Cash Flow/Total Debt CFFO/TD
    csq_f - Cost of Goods Sold/Inventory CGS/INV
    ac_f - Current Assets/Current Debt CA/CD
    as_f - Current Assets/Sales CA/S
    aa_f - Current Assets/Total Assets CA/TA
    cdtd_f - Current Debt/Total Debt CD/TD
    es_f - Earnings Before Interest & Taxes/Equity EBIT/EQ
    esa_f - Earnings Before Interest & Taxes/Sales EBIT/S
    eta_f - Earnings Betore Interest & Taxes/Total Assets EBIT/TA
    sa_f - Inventory/Current Assets INV/CA
    sw_f - Inventory/Working Capital INV/WC
    da_f - Long-Term Debt/Total Assets LTD/TA
    is_f - Net Income/Equity NI/EQ
    isa_f - Net Income/Sales NI/S
    ia_f - Net Income/Total Assets NI/TA
    ids_f - Net Income Plus Depreciation/Equity NIPD/EQ
    idsa_f - Net Income Plus Depreciation/Sales NIPD/S
    ida_f - Net Income Plus Depreciation/Total Assets NIPD/TA
    idl_f - Net Income Plus Depreciation/Total Debt NIPD/TD
    rs_f -  Receivables/Inventory REC/INV
    tdta_f - Total Debt/Total Assets TD/TA
    ws_f - Working Capital/Sales WC/S
    wta_f - Working Capital/Total Assets WC/TA
    wsq_f - Working Capital from Operations/Equity WCFO/EQ
    '''
   
    df['ccdc_f'] = df['cheq']/df['lctq']
    df['cs_f'] = df['cheq']/df['saleq']
    df['cta_f'] = df['cheq']/df['atq']
    df['cd_f'] = df['cheq']/df['ltq']
    df['cfe_f'] = df['oancfq']/df['seqq']
    df['cfta_f'] = df['oancfq']/df['atq']
    df['cftd_f'] = df['oancfq']/df['ltq']
    df['csq_f'] = df['cogsq']/df['saleq']
    df['ac_f'] = df['actq']/df['lctq']
    df['as_f'] = df['actq']/df['saleq']
    df['aa_f'] = df['actq']/df['atq']
    df['cdtd_f'] = df['lctq']/df['ltq']
    df['es_f'] = (df['ibq']+df['txtq']+df['xintq'])/df['seqq']
    df['esa_f'] = (df['ibq']+df['txtq']+df['xintq'])/df['saleq']
    df['eta_f'] = (df['ibq']+df['txtq']+df['xintq'])/df['atq']
    df['sa_f'] = df['saleq']/df['actq']
    df['sw_f'] = df['saleq']/(df['actq']+df['lctq'])
    df['da_f'] = df['dlttq']/df['atq']
    df['is_f'] = df['ibq']/df['seqq']
    df['isa_f'] = df['ibq']/df['saleq']
    df['ia_f'] = df['ibq']/df['atq']
    df['ids_f'] = (df['ibq']+df['dpq'])/df['seqq']
    df['idsa_f'] = (df['ibq']+df['dpq'])/df['saleq']
    df['ida_f'] = (df['ibq']+df['dpq'])/df['atq']
    df['idl_f'] = (df['ibq']+df['dpq'])/df['ltq']
    df['rs_f'] = df['rectq']/df['saleq']
    df['tdta_f'] = df['ltq']/df['atq']
    df['ws_f'] = (df['actq']+df['lctq'])/df['saleq']
    df['wta_f'] = (df['actq']+df['lctq'])/df['atq']
    df['wsq_f'] = (df['actq']+df['lctq'])/df['seqq']

    return df

def comp_second_features(df,startdate,enddate): 
    '''
    Compute summary statistic features from quarterly time series 
    of fundamental data and financial ratios.
    '''

    # cast and filter dataframe
    df['datadate'] = pd.to_datetime(df['datadate'],format='%Y%m%d')
    inputdf = df[(df.datadate >= startdate) & (df.datadate <= enddate)]

    # remove extra columns 
    tmpdf = inputdf[[col for col in inputdf.columns if "GICS" not in col and 'Unnamed' not in col]]  
    targdf = inputdf[[col for col in inputdf.columns if "GICS" not in col and 'Unnamed' not in col]]  

    # keep cusips with sufficient data, i.e. at least six quarters median value across 
    # all fields
    tmpdf = tmpdf.groupby('cusip').count().median(axis=1) 
    keep_cusips = tmpdf[tmpdf > 5.0].index  # keep at least 5 qtrs 

    inputdf = inputdf[inputdf.cusip.isin(keep_cusips)]

    xflds = [col for col in inputdf.columns if 'GICS' not in col]
    xflds = [col for col in xflds if col not in ['Unnamed: 0','Unnamed: 0.1']]
    
    ## keep datadata, cusip
    X = df[xflds]
    
    fdflst = [] 

    # Compute feature matrix
    for i,cusp in enumerate(keep_cusips): 
        if i%100. == 0.:
            print(i)
        tdf = X[X['cusip'] == cusp]
        tdf = tdf.set_index('datadate')
        tdf = tdf.drop('cusip',axis=1)
        
        ftrs = pd.concat([tdf.mean(),tdf.std(),tdf.pct_change().mean(),tdf.pct_change().std(),tdf.iloc[-1,:]],axis=1)

        ftrs.columns=['mn','std','chg_mn','chg_std','last_val'] 
        ftrs = ftrs.replace([np.inf,-np.inf],np.nan)

        ftrsfin = ftrs.stack()
        ftrsfin.index = ['_'.join(col) for col in ftrsfin.index] 

        fdflst.append(ftrsfin)

    fdf = pd.concat(fdflst,axis=1)
    fdf.columns=keep_cusips

    return fdf

def mod_run(X,y,num_leaves,n_estimators,add_name='',pltfigure=False):
    '''
    Main predictive model run where here 
    X -- predictor matrix  
    y -- target classification values
    '''
    # split data into 80% training 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=21)

    # split training data into 80% training and 20% validation 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=31)

    # Specify lgb classifier 
    gbm = lgb.LGBMClassifier(num_leaves=num_leaves,learning_rate=0.05,early_stopping_rounds=200,n_estimators=n_estimators)

    # fit the model on the training data using the validation set for performance
    # evaluation
    gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        early_stopping_rounds=5)

    # perform a final prediction on the hold out test set 
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    # Report performance metrics of classifier: 

    print("Balanced Accuracy")
    print(sm.balanced_accuracy_score(y_test,y_pred))

    labels = sorted(list(set(y_test)))

    print("Precision")
    for lab in labels:
        print(lab)
        print(sm.precision_score(y_test,y_pred,average=None,labels=[lab]))

    print("Micro Precision")
    print(sm.precision_score(y_test,y_pred,average='micro'))
    print("Macro Precision")
    print(sm.precision_score(y_test,y_pred,average='macro'))

    print("Recall")
    for lab in labels:
        print(lab)
        print(sm.recall_score(y_test,y_pred,average=None,labels=[lab]))
    print("Micro Recall") 
    print(sm.recall_score(y_test,y_pred,average='micro'))
    print("Macro Recall") 
    print(sm.recall_score(y_test,y_pred,average='macro'))

    print("F1")
    for lab in labels:
        print(lab)
        print(sm.f1_score(y_test,y_pred,average=None,labels=[lab]))

    # Report feature importances
    print('Feature importances:', list(gbm.feature_importances_))

    # Construct confusion matrix based on counts
    conmat = confusion_matrix(y_test.values,y_pred)
    seclabs = ['Energy','Matrls.','Indust.','Con.D.','Con.St.','Health','Fin.','IT','Comm.','Util.','Real E.']
    pltdf = pd.DataFrame(conmat,columns = seclabs,index = seclabs) 
    fconplt = pltdf

    if pltfigure:
        plt.figure(figsize=(25,20))
        sns.heatmap(fconplt,annot=True,cbar_kws={'label':'Count'},fmt='g')
        plt.title('GBT Confusion Matrix')
        plt.ylabel('Actual Categories')
        plt.xlabel('Classified Categories')
        plt.yticks(rotation=0)

        if True:
            plt.savefig(str(num_leaves) + '_' + str(n_estimators) + ' ' + add_name + '_IND.png')

    # Print accuracy:
    print('Accuracy')
    print(np.diag(conmat).sum()/conmat.sum().sum()) 

    return conmat,fconplt

if __name__ == "__main__": 
    ## Generate gics file as needed
    # gicsdf = combine_gics(write_flag=False)

    # Flag to either build mergedf.csv or read from file
    build = False
        
    targetcol = 'GICS_SECTOR'
    if build: 
        tmpdf = pd.read_csv('mergedf.csv',index_col=0)
        featdf = feature_build(tmpdf) 
        
        startdt = '2007-01-01' 
        enddt = '2017-01-01'
        X = comp_second_features(featdf,startdt,enddt).T

        cusipmap = tmpdf[['cusip',targetcol]].drop_duplicates()
        cusipmap = cusipmap.set_index('cusip')

        Xalt = X.join(cusipmap,how='left')
    else: 
        Xalt = pd.read_pickle('datafle.pkl')

    y = Xalt[targetcol]

    ## Values to study complexity/performance tradeoff of GBTs 
    nleaves = [2,3,4,5,8,10,12,16]
    nest = [25, 50, 75, 100, 125, 150, 175, 200]

    # Fix final 5 leave 200 iteration GBT model
    nleaves = [5]
    nest = [200]

    accmat = [] 

    for nst in nest: 
        acclst = []
        for lv in nleaves: 
            conmat, fconply = mod_run(Xalt.iloc[:,:-1],Xalt.iloc[:,-1],lv,nst,add_name='',pltfigure=True)
            acc = np.diag(conmat).sum()/conmat.sum().sum()
            acclst.append(acc)
        accmat.append(acclst)
    
    # save results
    pd.DataFrame(accmat,index=nest,columns=nleaves).to_csv('accmat.csv')

    ### Come back later and alter to consider rolling studies 
    if False:
        n_leaves = [16]
        n_est = [200]
        start_date = '2015-01-01' 
        end_date = '2016-01-01'

        tmpdf = pd.read_csv('mergedf.csv',index_col=0)
        featdf = feature_build(tmpdf) 

        endofqtr = ['03-31','06-30','09-30','12-31'] 
        yrlst = np.arange(1992,2018).astype(int)
        qtrdtes = [pd.to_datetime(str(yr) + '-' + qr) for yr in yrlst for qr in endofqtr]
    
        for i in range(len(qtrdtes)-4):
            tmp1 = tmpdf['datadate'] 
            num_comps,secacc,accy,cols = timeres(qtrdtes[i],qtrdtes[i+4],target_col,num_leaves,num_est)
            num_lst.append(num_comps)
            acc_lst.append(accy)
            sec_lst.append(secacc)

        resdf = pd.DataFrame(np.array([num_lst,acc_lst]).T,index=qtrdtes[4:],columns=['num_comp','acc']) 
        secdf = pd.DataFrame(sec_lst,columns=cols)

        #resdf.to_csv('res_' + str(num_leaves) + '_' + str(num_est) + '.csv')
        #secdf.to_csv('sec_' + str(num_leaves) + '_' + str(num_est) + '.csv')
