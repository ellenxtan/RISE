import os
import random
import pandas as pd
import numpy as np

## https://developer.apple.com/metal/tensorflow-plugin/
## /Users/xtan/miniforge3/bin/python3
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import export_text

from sklearn.model_selection import GridSearchCV

import collections

# pd.options.mode.chained_assignment = None  # default='warn'

# Keras and TensorFlow options
layers = [2,3]
nodes=[256,512,1024]
dropouts=[0.2] #,0.4
acts = ["sigmoid","relu"]
opts = ["adam","nadam"]
bsizes = [32,64] #,128
n_epochs = [50,100] #,200


def set_random(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    return


def normalize(df_tr, df_te, names_to_norm):
    df = df_tr.copy()
    df_test = df_te.copy()

    # normalize X & S
    scaler = StandardScaler()
    df[names_to_norm] = scaler.fit_transform(df[names_to_norm])
    df_test[names_to_norm] = scaler.transform(df_test[names_to_norm])

    return(df.copy(), df_test.copy())


def get_ps_fit(df_train, use_covars): #, mod, is_tune
    df_tr = df_train.copy()

    #TODO: tune PS fit, right now only rf with default params
    # params = {'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 100}
    # regr = RandomForestClassifier(random_state=1234, 
    #                                 n_estimators=params["n_estimators"], 
    #                                 min_samples_split=params["min_samples_split"], 
    #                                 max_features=params["max_features"],
    #                                 ).fit(df_tr[use_covars], df_tr["A"])

    regr = RandomForestClassifier(random_state=1234).fit(df_tr[use_covars], df_tr["A"])
    
    return(regr)


def keras_wrap(x_train, train_labels, train_wts, x_test, loss_fn, act_out, 
               layer=2, node=1024, dropout=0.2, n_epoch=100, bsize=64, act="relu", 
               opt="Adam", val_split=0.2, is_early_stop=True, verb=0):

    if is_early_stop:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        callback = [early_stop]
    else:
        callback = None

    # set input_dim for the number of features
    if len(x_train.shape) == 1:
        input_dim = 1
    else:
        input_dim = x_train.shape[1]
    
    # model
    model = Sequential()
    for i in range(layer):
        if i==0:
            model.add(Dense(node, input_dim=input_dim, activation=act)) # Hidden 1
            model.add(Dropout(dropout))
        else:
           model.add(Dense(node, activation=act)) # Hidden 2 
           model.add(Dropout(dropout))
           
    model.add(Dense(1, activation=act_out)) # Output
    
    model.compile(loss=loss_fn, optimizer=opt)
    model.fit(x_train, train_labels, 
              sample_weight=train_wts,
              epochs=n_epoch, batch_size=bsize,
              validation_split=val_split, callbacks=callback, verbose=verb)
    
    # predict
    pred_test = model.predict(x_test).flatten()
    pred_train = model.predict(x_train).flatten()
    return pred_test, pred_train, model


def hyper_tuning(x_train, train_labels, train_wts, loss_fn, act_out,
                 layers, nodes, dropouts, acts, opts, bsizes, n_epochs, 
                 n_cv=5, n_jobs=1):
    """
    layers = [2,3]
    nodes=[100,300,512]
    dropout=[0.2] #,0.4
    activation = ["sigmoid","relu"]
    optimizer = ["adam"] #,"nadam"
    bsize = [32,64] #,128
    n_epochs = [50,100] #,200

    bst_params = hyper_tuning(x_train, train_labels, train_wts, loss_fn, act_out,
                              layers, nodes, dropouts, acts, opts, bsizes, n_epochs, 
                              n_cv=5, n_jobs=1)
    """
    # set input_dim for the number of features
    if len(x_train.shape) == 1:
        input_dim = 1
    else:
        input_dim = x_train.shape[1]
    
    def create_model(layers,nodes,acts,opts,dropouts):
        model = Sequential()
        for i in range(layers):
            if i==0:
                model.add(Dense(nodes, input_dim=input_dim))
                model.add(Activation(acts))
                model.add(Dropout(dropouts))
            else:
                model.add(Dense(nodes, activation=acts)) 
                model.add(Activation(acts))
                model.add(Dropout(dropouts))
    
        model.add(Dense(units=1, activation=act_out))
        model.compile(optimizer=opts, loss=loss_fn)
        return model

    if act_out == "sigmoid": #for classification
        model = KerasClassifier(build_fn=create_model, verbose=2)
    else: #None #for regression including quantile
        model = KerasRegressor(build_fn=create_model, verbose=2)
    
    param_grid = dict(layers=layers, nodes=nodes, acts=acts, opts=opts, 
                      dropouts=dropouts, bsizes=bsizes, n_epochs=n_epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_cv, n_jobs=n_jobs)
    
    grid_result = grid.fit(x_train, train_labels, sample_weight=train_wts)
    
    print("hyperparams tuning:", grid_result.best_score_,grid_result.best_params_)

    bst_params = grid_result.best_params_
    layer = bst_params['layers']
    node = bst_params['nodes']
    dropout = bst_params['dropouts']
    n_epoch = bst_params['n_epochs']
    bsize = bst_params['bsizes']
    act = bst_params['acts']
    opt = bst_params['opts']

    return layer, node, dropout, n_epoch, bsize, act, opt


def fit_expectation(obj_name, df, use_covars, df_pred, is_tune, is_class, df_val=None):
    """E(Y|X=x,A=a) or E(Y|X=x,S=s,A=a)
    (model separated by A)
    """
    df_tr = df.copy()
    df_te = df_pred.copy()
    if df_val is not None:
        df_va = df_val.copy()

    # fit on df0/df1
    if is_class:
        loss_fn = 'binary_crossentropy'
        act_out = 'sigmoid'
    else:
        loss_fn = "mean_squared_error"
        act_out = None
    
    if is_tune:
        layer, node, dropout, n_epoch, bsize, act, opt = \
                hyper_tuning(df_tr[use_covars], df_tr["Y"], None, loss_fn, act_out,
                    layers, nodes, dropouts, n_epochs, bsizes, acts, opts, 
                    n_cv=5, n_jobs=1)
        Yhat, _, regr = keras_wrap(df_tr[use_covars], df_tr["Y"], None, 
                            df_te[use_covars], loss_fn, act_out, 
                            layer, node, dropout, n_epoch, bsize, act, opt, 
                            val_split=None, is_early_stop=False, verb=0)
    else:
        Yhat, _, regr = keras_wrap(df_tr[use_covars], df_tr["Y"], None, 
                            df_te[use_covars], loss_fn, act_out, 
                            layer=2, node=1024, dropout=0.2, n_epoch=100, bsize=64, act="relu", opt="Adam", 
                            val_split=0.2, is_early_stop=True, verb=0)
        # Yhat, _, regr = keras_expectation(df_tr[use_covars], df_tr["Y"], df_te[use_covars], loss)

    if df_val is not None:
        if not is_class: #NRMSE
            y_tr = regr.predict(df_tr[use_covars])
            rms = mean_squared_error(df_tr['Y'], y_tr, squared=False)
            met = rms / (np.max(df_tr['Y']) - np.min(df_tr['Y']))
            print(obj_name, "evaluate on train: NRMSE", met)

            y_va = regr.predict(df_va[use_covars])
            rms = mean_squared_error(df_va['Y'], y_va, squared=False)
            met = rms / (np.max(df_va['Y']) - np.min(df_va['Y']))
            print(obj_name, "evaluate on test : NRMSE", met)
        elif is_class: #AUC
            y_tr = regr.predict(df_tr[use_covars])
            met = roc_auc_score(df_tr['Y'], y_tr)
            print(obj_name, "evaluate on train: AUC", met)

            y_va = regr.predict(df_va[use_covars])
            met = roc_auc_score(df_va['Y'], y_va)
            print(obj_name, "evaluate on test : AUC", met)

    return(Yhat, regr)


def fit_quantile(x_train, train_labels, x_test, q, is_tune):
    """
    quantile regression: yhat | X,A
    """
    def tilted_loss(q, y, f):
        """quantile loss for Keras"""
        e = (y - f)
        return keras.backend.mean(keras.backend.maximum(q * e, (q - 1) * e), axis=-1)

    loss_fn = lambda y, f: tilted_loss(q, y, f)
    act_out = None

    if is_tune:
        layer, node, dropout, n_epoch, bsize, act, opt = \
                    hyper_tuning(x_train, train_labels, None, loss_fn, act_out,
                        layers, nodes, dropouts, n_epochs, bsizes, acts, opts, 
                        n_cv=5, n_jobs=1)
        pred_test, _, model = keras_wrap(x_train, train_labels, None, 
                        x_test, loss_fn, act_out, 
                        layer, node, dropout, n_epoch, bsize, act, opt, 
                        val_split=None, is_early_stop=False, verb=0)
    else:
        pred_test, _, model = keras_wrap(x_train, train_labels, None, x_test, loss_fn, act_out, 
                layer=2, node=1024, dropout=0.2, n_epoch=100, bsize=64, act="relu", 
                opt="Adam", val_split=0.2, is_early_stop=True, verb=0)
        # fit_qua, preds = keras_quantile(x_train, train_labels, x_test, q)

    return model, pred_test 


def fit_infinite(df, X_names, XS_names, fit):
    df_use = df.copy()

    S_names = list( set(XS_names) - set(X_names) )

    if len(S_names) == 1:
        s_min = int(np.min(df_use["S"]))
        s_max = int(np.max(df_use["S"]))
        s_grid = range(s_min, s_max+1)
    else: #"sepsis-mul-cat"
        s_lst = []
        for s_var in S_names:
            s_min = int(np.min(df_use[s_var]))
            s_max = int(np.max(df_use[s_var]))
            s_grid = range(s_min, s_max+1)
            s_lst.append(np.array(s_grid))
        s_grid = s_lst.copy()
        print("s_grid:", s_grid) #2401

    X_df = df_use[X_names].copy()
    idx = range(X_df.shape[0])
    X_df["idx"] = idx

    if len(S_names) == 1:
        idx_s = np.array(np.meshgrid(idx, s_grid)).reshape(2, len(idx)*len(s_grid)).T
        idx_s = pd.DataFrame(idx_s, columns = ['idx','S'])
    else:
        idx_s_lst = [np.array(idx)] + s_grid
        idx_s = np.array(np.meshgrid(*np.array(idx_s_lst, dtype=object))).reshape(1+len(S_names), -1).T
        idx_s = pd.DataFrame(idx_s, columns = ['idx']+S_names)

    aug_df = X_df.merge(idx_s, on='idx', how='left').copy()

    # predict Y
    aug_df["pred"] = fit.predict(aug_df[XS_names])
    
    smry = aug_df.groupby(by="idx", as_index=False).agg({"pred":["min"]})

    return(smry.loc[:,("pred","min")].values.copy())


def get_label_wt(g1, g0):
    """get label & weight for classification"""
    c1 = g1
    c2 = g0
    label = np.sign(c1 - c2)
    label = np.where(label==1, 1, 0)
    # label_cl = np.where(label==1, "yes","no")
    weight = np.abs(c1 - c2)
    # weight = weight / np.sum(weight)

    return(label, weight)


def class_pred(df_train, df_test, use_covars, label, weight, is_tune, s_type):
    """binary classification with sample weights for decision rule by Keras
    """
    df_tr = df_train.copy()
    df_te = df_test.copy()

    loss_fn = 'binary_crossentropy'
    act_out = 'sigmoid'

    # classification (train & pred)
    if is_tune:
        layer, node, dropout, n_epoch, bsize, act, opt = \
                    hyper_tuning(df_tr[use_covars], label, weight, loss_fn, act_out,
                        layers, nodes, dropouts, n_epochs, bsizes, acts, opts, 
                        n_cv=5, n_jobs=1)
        prob_test, prob_train, model = keras_wrap(df_tr[use_covars], label, weight, 
                        df_te[use_covars], loss_fn, act_out, 
                        layer, node, dropout, n_epoch, bsize, act, opt, 
                        val_split=None, is_early_stop=False, verb=0)
    else:
        prob_test, prob_train, model = keras_wrap(df_tr[use_covars], label, weight, 
                        df_te[use_covars], loss_fn, act_out, 
                        layer=2, node=1024, dropout=0.2, n_epoch=100, bsize=64, act="relu", 
                        opt="Adam", val_split=0.2, is_early_stop=True, verb=0)
        # prob_test, prob_train, model = keras_pred(x_train=df_tr[use_covars], train_labels=label, 
        #             train_wts=weight, x_test=df_te[use_covars])
    
    # print(prob_test)
    pred_test = np.where(prob_test>0.5, "yes", "no")
    pred_train = np.where(prob_train>0.5, "yes", "no")
        
    # print table
    print("pred_train table:", collections.Counter(pred_train.ravel()))
    print("pred_test table:", collections.Counter(pred_test.ravel()))
    if s_type == "disc": #disc S
        df_tr["pred_train"] = pred_train
        df_te["pred_test"] = pred_test
        print(pd.crosstab(df_tr["S"], df_tr["pred_train"], normalize='index'))
        print(pd.crosstab(df_te["S"], df_te["pred_test"], normalize='index'))
    print("="*30)

    return(pred_test, pred_train, model)


def optimal_AS_inf(df, proj, is_sim, s_type, X_names, XS_names, y_fn, 
                   is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1):
    """
    Used for real data with disc S | simulation with disc/cont S

    get df with minority index
        input:  df_train for E(Y|X,S,A) and df_test for Yhat -> optimal S
        output: df_test with vulnerable index
    """
    df_use = df.copy()

    # S grid from empirical test data
    a_grid = range(0, 2)
    if s_type == "disc": #TODO: multi-category
        s_grid = range(0, 2)
        assert len(s_grid) == 2
    elif s_type == "cont":
        s_grid = df_use["S"].sort_values().values.copy() #TODO: multiple S
        assert len(s_grid) > 2
    else:
        assert(1 == 0)
    print("len(s_grid):", len(s_grid)) #2
    
    # augment X with A&S grid
    a_s_grid = np.array(np.meshgrid(a_grid, s_grid)).reshape(2, len(a_grid)*len(s_grid)).T
    a_s_grid = pd.DataFrame(a_s_grid, columns = ['A','S'])
    idx_as = range(a_s_grid.shape[0])
    a_s_grid["idx_as"] = idx_as #4802=2401*2

    X_df = df_use[X_names].copy()
    idx_i = range(X_df.shape[0])
    X_df["idx_i"] = idx_i #2401

    idxs = np.array(np.meshgrid(idx_as, idx_i)).reshape(2, len(idx_as)*len(idx_i)).T
    idxs = pd.DataFrame(idxs, columns = ['idx_as','idx_i'])

    aug_df = X_df.merge(idxs, on='idx_i', how='left').copy()
    aug_df = aug_df.merge(a_s_grid, on='idx_as', how='left').copy() #11529602=2401*4802

    # aug_df: each X with all combinations of (S,A)
    aug_df = aug_df[['idx_i','A','S']+X_names].copy()
    assert(aug_df.shape[0] == len(a_grid)*len(s_grid)*df_use.shape[0])
    aug_df_a0 = aug_df[aug_df["A"]==0].reset_index(drop=True).copy() #5764801=11529602/2
    aug_df_a1 = aug_df[aug_df["A"]==1].reset_index(drop=True).copy() #5764801=11529602/2

    # get E[Y|X,S,A=d(x)]
    if is_sim: # from true distn (simulation)
        aug_df_a0["EY"] = aug_df_a0.eval(y_fn)
        aug_df_a1["EY"] = aug_df_a1.eval(y_fn)
    else: # from fitted model (real data)
        if proj == "sepsis": # fix S vars except lactate at mean (only vary lactate)
            for col in XS_names:
                if col not in aug_df_a0.columns:
                    aug_df_a0[col] = df_use[df_use['A']==0][col].mean()
                    aug_df_a1[col] = df_use[df_use['A']==1][col].mean()

        aug_df_a0["EY"] = fit_Y_XS_A0.predict(aug_df_a0[XS_names])
        aug_df_a1["EY"] = fit_Y_XS_A1.predict(aug_df_a1[XS_names])

    # find quantile/inf wrt S
    def quan(x):
        return x.quantile(qua_use)

    if s_type == "disc":
        agg_fn1 = {"EY":["min"]}
        agg_fn3 = {"EY_min":"min"}
    else:
        agg_fn1 = {"EY":[quan]}
        agg_fn3 = {"EY_quan":"min"}

    # min E(Y|X,S,A) separate for A=0 and A=1
    min_a0 = aug_df_a0.groupby(by="idx_i", as_index=False).agg(agg_fn1)
    min_a0.columns = ["idx_i"] + ['_'.join(col) for col in min_a0.columns[1:]]
    aug_df_a0_min = aug_df_a0.merge(min_a0, on='idx_i').copy()
    min_a1 = aug_df_a1.groupby(by="idx_i", as_index=False).agg(agg_fn1)
    min_a1.columns = ["idx_i"] + ['_'.join(col) for col in min_a1.columns[1:]]
    aug_df_a1_min = aug_df_a1.merge(min_a1, on='idx_i').copy()
    aug_df_cat = pd.concat([aug_df_a0_min, aug_df_a1_min]).reset_index(drop=True).copy()

    # find S*: min-min E(Y|X,S,A)
    min_df = aug_df_cat.groupby(by="idx_i", as_index=False).agg(agg_fn3).copy()
    min_min_df = aug_df_cat.merge(min_df, on='idx_i',suffixes=('', '_min'))

    # get S_opt (S*) vulnerable groups
    if s_type == "disc":
        msk1 = min_min_df['EY_min'] == min_min_df['EY_min_min']
        msk2 = min_min_df['EY'] <= min_min_df['EY_min_min']
    elif s_type == "cont":
        msk1 = min_min_df['EY_quan'] == min_min_df['EY_quan_min']
        msk2 = min_min_df['EY'] <= min_min_df['EY_quan_min']
    df_filter = min_min_df[msk1 & msk2].reset_index(drop=True).copy() # may contains more rows than df_use
    assert( df_filter.shape[0] >= df_use.shape[0] )

    X_df = df_use[X_names].copy()
    X_df["idx_i"] = idx_i
    df_S_opt = X_df.merge(df_filter, on=['idx_i']+X_names, how="left").copy()
    df_S_opt.rename(columns={"S":"S_opt"}, inplace=True)
    # df_S_opt should be used to plot tree (find pattern of X->S)

    # link S* to observed data
    XS_df = df_use[X_names+['S']].copy()
    XS_df["idx_i"] = idx_i
    merge_df = XS_df.merge(df_S_opt, left_on=['idx_i','S']+X_names, right_on=['idx_i','S_opt']+X_names, how="left").copy()
    merge_df = merge_df.drop_duplicates(subset=['idx_i']+X_names, keep='first').reset_index(drop=True).copy()
    
    S_opt = merge_df['S_opt'].copy() # if not NaN, this subject is vulnerable
    assert( S_opt.shape[0] == X_df.shape[0] )

    # minor_index: 1-vulnerable, 0-rest
    merge_df['M_opt'] = np.where(merge_df['S']==merge_df['S_opt'], 1, 0)
    print("M_opt 1:vulnerable 0:rest\n", merge_df['M_opt'].value_counts(dropna=False))
    M_opt = merge_df['M_opt'].copy()
    assert( M_opt.shape[0] == X_df.shape[0] )
    
    return M_opt # valid_df should be used to plot tree (find pattern of X->S) #A_opt, S_opt, , df_S_opt


def optimal_AS_inf_mulS(df, proj, is_sim, s_type, X_names, XS_names, y_fn, 
                        is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1):
    """
    Used for real data with disc S | simulation with disc/cont S

    get df with minority index
        input:  df_train for E(Y|X,S,A) and df_test for Yhat -> optimal A and S
        output: df_test with optimal A and S
    """
    df_use = df.copy()

    # A grid
    a_grid = range(0, 2)
    assert s_type == "disc"

    # S grid from empirical test data
    S_names = list( set(XS_names) - set(X_names) )
    s_lst = []
    for s_var in S_names:
        s_min = int(np.min(df_use[s_var]))
        s_max = int(np.max(df_use[s_var]))
        s_grid = range(s_min, s_max+1)
        s_lst.append(np.array(s_grid))
    s_grid = s_lst.copy()
    print("s_grid:", s_grid) #2401
    
    # augment X with A&S grid
    a_s_lst = [np.array(a_grid)] + s_grid
    a_s_grid = np.array(np.meshgrid(*np.array(a_s_lst))).reshape(1+len(S_names), -1).T
    a_s_grid = pd.DataFrame(a_s_grid, columns = ['A']+S_names)
    idx_as = range(a_s_grid.shape[0])
    a_s_grid["idx_as"] = idx_as #4802=2401*2

    X_df = df_use[X_names].copy()
    idx_i = range(X_df.shape[0])
    X_df["idx_i"] = idx_i #2401

    idxs = np.array(np.meshgrid(idx_as, idx_i)).reshape(2, len(idx_as)*len(idx_i)).T
    idxs = pd.DataFrame(idxs, columns = ['idx_as','idx_i'])

    aug_df = X_df.merge(idxs, on='idx_i', how='left').copy()
    aug_df = aug_df.merge(a_s_grid, on='idx_as', how='left').copy() #11529602=2401*4802

    # aug_df: each X with all combinations of (S,A)
    aug_df = aug_df[['idx_i','A']+XS_names].copy()
    assert(aug_df.shape[0] == len(a_grid)*len(s_grid)**len(S_names)*df_use.shape[0])
    aug_df_a0 = aug_df[aug_df["A"]==0].reset_index(drop=True).copy() #5764801=11529602/2
    aug_df_a1 = aug_df[aug_df["A"]==1].reset_index(drop=True).copy() #5764801=11529602/2

    # get E[Y|X,S,A=d(x)]
    if is_sim: # from true distn (simulation)
        aug_df_a0["EY"] = aug_df_a0.eval(y_fn)
        aug_df_a1["EY"] = aug_df_a1.eval(y_fn)
    else: # from fitted model (real data) #TODO: xgboost
        if proj == "sepsis": # fix S vars except lactate at mean (only vary lactate)
            for col in XS_names:
                if col not in aug_df_a0.columns:
                    aug_df_a0[col] = df_use[df_use['A']==0][col].mean()
                    aug_df_a1[col] = df_use[df_use['A']==1][col].mean()

        aug_df_a0["EY"] = fit_Y_XS_A0.predict(aug_df_a0[XS_names])
        aug_df_a1["EY"] = fit_Y_XS_A1.predict(aug_df_a1[XS_names])

    # find quantile/inf wrt S
    def quan(x):
        return x.quantile(qua_use)

    if s_type == "disc":
        agg_fn1 = {"EY":["min"]}
        agg_fn3 = {"EY_min":"min"}
    else:
        agg_fn1 = {"EY":[quan]}
        agg_fn3 = {"EY_quan":"min"} # TODO: should it be lower/upper quantile???? --yes?

    # min E(Y|X,S,A) separate for A=0 and A=1
    min_a0 = aug_df_a0.groupby(by="idx_i", as_index=False).agg(agg_fn1)
    min_a0.columns = ["idx_i"] + ['_'.join(col) for col in min_a0.columns[1:]]
    aug_df_a0_min = aug_df_a0.merge(min_a0, on='idx_i').copy()
    min_a1 = aug_df_a1.groupby(by="idx_i", as_index=False).agg(agg_fn1)
    min_a1.columns = ["idx_i"] + ['_'.join(col) for col in min_a1.columns[1:]]
    aug_df_a1_min = aug_df_a1.merge(min_a1, on='idx_i').copy()
    aug_df_cat = pd.concat([aug_df_a0_min, aug_df_a1_min]).reset_index(drop=True).copy()

    # find S*: min-min E(Y|X,S,A)
    min_df = aug_df_cat.groupby(by="idx_i", as_index=False).agg(agg_fn3).copy()
    min_min_df = aug_df_cat.merge(min_df, on='idx_i',suffixes=('', '_min'))

    # get S_opt (S*) vulnerable groups
    if s_type == "disc":
        msk1 = min_min_df['EY_min'] == min_min_df['EY_min_min']
        msk2 = min_min_df['EY'] <= min_min_df['EY_min_min']
    elif s_type == "cont":
        msk1 = min_min_df['EY_quan'] == min_min_df['EY_quan_min']
        msk2 = min_min_df['EY'] <= min_min_df['EY_quan_min']
    df_filter = min_min_df[msk1 & msk2].reset_index(drop=True).copy() # may contains more rows than df_use
    assert( df_filter.shape[0] >= df_use.shape[0] )

    X_df = df_use[X_names].copy()
    X_df["idx_i"] = idx_i
    df_S_opt = X_df.merge(df_filter, on=['idx_i']+X_names, how="left").copy()
    # df_S_opt.rename(columns={"S":"S_opt"}, inplace=True)
    df_S_opt.columns = [str(col) + '_opt' if col in S_names else str(col) for col in df_S_opt.columns]
    # df_S_opt should be used to plot tree (find pattern of X->S)
    
    # link S* to observed data
    XS_df = df_use[XS_names].copy()
    XS_df["idx_i"] = idx_i
    S_opt_names = [str(col) + '_opt' for col in S_names]
    merge_df = XS_df.merge(df_S_opt, left_on=['idx_i']+X_names+S_names, right_on=['idx_i']+X_names+S_opt_names, how="left").copy()
    merge_df = merge_df.drop_duplicates(subset=['idx_i']+X_names, keep='first').reset_index(drop=True).copy()
    
    S_opt = merge_df['S_opt'].copy() # if not NaN, this subject is vulnerable
    assert( S_opt.shape[0] == X_df.shape[0] )

    # minor_index
    merge_df['M_opt'] = np.where(merge_df['S']==merge_df['S_opt'], 1, 0)
    print("M_opt 1:minor 0:rest\n", merge_df['M_opt'].value_counts(dropna=False))
    M_opt = merge_df['M_opt'].copy()
    assert( M_opt.shape[0] == X_df.shape[0] )

    return M_opt # valid_df should be used to plot tree (find pattern of X->S) #A_opt, S_opt, , df_S_opt


def minor_rule(df, use_covars, s_type, S_name):
    """use decision tree to print out rule of vulnerable group (X -> S)"""
    df_use = df.copy()

    df_use["S_minor"] = np.where(np.isnan(df_use[S_name]), -999,df_use[S_name]) #-100 means tie
    print('df_use["S_minor"]', df_use["S_minor"].value_counts(dropna=False))

    if s_type == "disc":
        clf = DecisionTreeClassifier(max_depth=3, random_state=4211) #criterion="entropy", 
    else:
        clf = DecisionTreeRegressor(max_depth=3, random_state=4211) #max_depth=3, 
    
    # clf.fit(df_use[use_covars], df_use["sgn"])
    clf.fit(df_use[use_covars], df_use["S_minor"])
    
    # export the decision rules
    tree_rules = export_text(clf, feature_names = list(use_covars))
    print(S_name)
    print(tree_rules)

    clf.predict(df_use[use_covars])

    return


def expect_reward(df, A_name, minor_name): #, XS_names, pr_a1, fit_ps
    df_use = df.copy()
    
    minor_df = df_use[df_use[minor_name]==1].reset_index(drop=True).copy()
    rest_df = df_use[df_use[minor_name]==0].reset_index(drop=True).copy()

    numer = df_use["Y"] * (df_use["A_obs"] == df_use[A_name]) / df_use["ps"]
    denom = (df_use["A_obs"] == df_use[A_name]) / df_use["ps"]
    reward_all = np.sum(numer) / np.sum(denom)

    if not minor_df.empty:
        numer = minor_df["Y"] * (minor_df["A_obs"] == minor_df[A_name]) / minor_df["ps"]
        denom = (minor_df["A_obs"] == minor_df[A_name]) / minor_df["ps"]
        reward_minor = np.sum(numer) / np.sum(denom)
    else:
        reward_minor = np.nan

    if not rest_df.empty:
        numer = rest_df["Y"] * (rest_df["A_obs"] == rest_df[A_name]) / rest_df["ps"]
        denom = (rest_df["A_obs"] == rest_df[A_name]) / rest_df["ps"]
        reward_rest = np.sum(numer) / np.sum(denom)
    else:
        reward_rest = np.nan

    print(A_name, "reward  ", 
          "all:",round(reward_all,3), 
          "minor:",round(reward_minor,3),
          "rest:",round(reward_rest,3))

    return reward_all, reward_minor, reward_rest


def get_pred_Y(df, A_name, XS_names, y_fn, model_sxa0, model_sxa1):
    """generate Y based on predicted A (treatment assignment)
    model: Y|X,S,A=a
    should use unnormalized X & S for simulation & normalized for real data!
    """
    df_use = df[XS_names].copy()

    df_use["A"] = np.where(df[A_name]=="yes", 1,0)
    if y_fn is not None:
        Y_pred = df_use.eval(y_fn)
    else:
        Y_pred0 = model_sxa0.predict(df_use[XS_names]).flatten()
        Y_pred1 = model_sxa1.predict(df_use[XS_names]).flatten()
        Y_pred = np.where(df_use["A"]==0, Y_pred0, Y_pred1)
    assert len(Y_pred.shape)==1

    return Y_pred


def get_value_fn(df, A_name, XS_names, minor_name, y_fn, model_sxa0, model_sxa1):
    """get value function for subgroup of interest"""
    df_use = df.copy()

    if 'S_err' in XS_names:
        X_names = list(df_use.columns[np.flatnonzero(np.char.startswith(list(df_use.columns), 'X'))])
        XS_names = X_names + ['S']

    df_use["Y_pred"] = get_pred_Y(df_use, A_name, XS_names, y_fn, model_sxa0, model_sxa1)

    minor_df = df_use[df_use[minor_name]==1].reset_index(drop=True).copy()
    rest_df = df_use[df_use[minor_name]==0].reset_index(drop=True).copy()

    value_all = np.mean(df_use["Y_pred"].values)

    if not minor_df.empty:
        value_minor = np.mean(minor_df["Y_pred"].values)
    else:
        value_minor = np.nan
    
    if not rest_df.empty:
        value_rest = np.mean(rest_df["Y_pred"].values)
    else:
        value_rest = np.nan
    
    print(A_name, "value   ", 
          "all:",round(value_all,3), 
          "minor:",round(value_minor,3),
          "rest:",round(value_rest,3))

    return value_all, value_minor, value_rest


def get_obj_qua(df, X_names, s_type, A_name, minor_name, fit_qua0, fit_qua1):
    """
    Used for real data with cont S (no need S grid)
    
    df includes X,A_test (A_test is from prediction)
    objective: E[Gs{E[Y|X,S,A=d(x)]}]
    -> Quantile_regress Y|X,A -> E[.]
    -> take avg over all subjects
    """
    df_use = df.copy()
    del df_use['A']
    del df_use['Y']

    if s_type != "cont":
        assert(1 == 0)

    Y_pred0 = fit_qua0.predict(df_use[X_names]).flatten()
    Y_pred1 = fit_qua1.predict(df_use[X_names]).flatten()

    df_use["A"] = np.where(df_use[A_name]=="yes", 1,0) #contains X & A
    df_use["Y"] = np.where(df_use["A"]==0, Y_pred0, Y_pred1)

    minor_df = df_use[df_use[minor_name]==1].reset_index(drop=True).copy()
    rest_df = df_use[df_use[minor_name]==0].reset_index(drop=True).copy()

    obj_all = np.mean(df_use["Y"].values)

    if not minor_df.empty:
        obj_minor = np.mean(minor_df["Y"].values)
    else:
        obj_minor = np.nan
    
    if not rest_df.empty:
        obj_rest = np.mean(rest_df["Y"].values)
    else:
        obj_rest = np.nan
    
    print(A_name, "obj     ", 
          "all:",round(obj_all,3), 
          "minor:",round(obj_minor,3),
          "rest:",round(obj_rest,3))
    
    return obj_all, obj_minor, obj_rest


def get_obj_inf(df, X_names, XS_names, A_name, minor_name, y_fn, s_type, is_class, qua_use, model_sxa0, model_sxa1):
    """
    Used for real data with disc S | simulation with disc/cont S

    df includes X,A_test (A_test is from prediction)
    objective: E[Gs{E[Y|X,S,A=d(x)]}]
    -> get E[Y|X,S,A=d(x)] from true distn (simulation) or fitted model (real data)
    -> given X & A, generate S from empirical test data
    -> find quan/inf wrt S 
    -> take avg over all subjects
    """
    df_use = df.copy()

    # S grid from empirical test data
    if s_type == "disc": # real/simulation with disc S
        S_names = list( set(XS_names) - set(X_names) )
        if len(S_names) == 1:
            s_min = int(np.min(df_use["S"]))
            s_max = int(np.max(df_use["S"]))
            s_grid = range(s_min, s_max+1)
            if y_fn is None:
                s_grid = [s_grid]
        else: #"sepsis-mul-cat"
            s_lst = []
            for s_var in S_names:
                s_min = int(np.min(df_use[s_var]))
                s_max = int(np.max(df_use[s_var]))
                s_grid = range(s_min, s_max+1)
                s_lst.append(np.array(s_grid))
            s_grid = s_lst.copy()
            # print("s_grid:", s_grid) #2401
    elif s_type == "cont" and y_fn is not None: # simulation with cont S
        s_grid = df["S"].values.copy() #TODO: multiple S (no need for now as no simulation has multiple S)
        assert len(s_grid) > 2
        # s_grid = [s_grid]
        # S_names = ['S'] # caveat: for measurement error case
    else:
        assert(1 == 0)
    # print("len(s_grid):", len(s_grid))

    # augment X&A with S grid
    df_XA = df_use[[minor_name]+X_names].copy() #contains X
    df_XA["A"] = np.where(df_use[A_name]=="yes", 1,0) #contains X & A
    idx = range(df_XA.shape[0])
    df_XA["idx"] = idx

    if y_fn is None:
        idx_s_lst = [np.array(idx)] + s_grid
        idx_s = np.array(np.meshgrid(*np.array(idx_s_lst, dtype=object))).reshape(1+len(S_names), -1).T
        idx_s = pd.DataFrame(idx_s, columns = ['idx']+S_names)
    else: # if remove, may occur error in .eval(y_fn) # this works as long as there is no multi-S in simulation
        idx_s = np.array(np.meshgrid(idx, s_grid)).reshape(2, len(idx)*len(s_grid)).T
        idx_s = pd.DataFrame(idx_s, columns = ['idx','S'])

    aug_df = df_XA.merge(idx_s, on='idx', how='left').reset_index(drop=True).copy()

    # get E[Y|X,S,A=d(x)]
    if y_fn is not None: # from true distn (simulation)
        aug_df["Y"] = aug_df.eval(y_fn)
    else: # from fitted model (real data)
        Y_pred0 = model_sxa0.predict(aug_df[XS_names]).flatten()
        Y_pred1 = model_sxa1.predict(aug_df[XS_names]).flatten()
        aug_df["Y"] = np.where(aug_df["A"]==0, Y_pred0, Y_pred1)

    # find quantile/inf wrt S
    def quan(x):
        return x.quantile(qua_use)
    
    minor_df = aug_df[aug_df[minor_name]==1].reset_index(drop=True).copy()
    rest_df = aug_df[aug_df[minor_name]==0].reset_index(drop=True).copy()

    # Gs{E[Y|X,S,A=d(x)]}
    if (s_type == "cont"):
        agg_fn = {"Y": quan} #"mean","min",
    elif (s_type == "disc"):
        agg_fn = {"Y": "min"} #"mean",quan,

    smry_all = aug_df.groupby(by="idx", as_index=False).agg(agg_fn)
    smry_minor = minor_df.groupby(by="idx", as_index=False).agg(agg_fn)
    smry_rest = rest_df.groupby(by="idx", as_index=False).agg(agg_fn)

    # obj: E{G(.)}
    obj_all = np.mean(smry_all.loc[:,"Y"].values)

    if not minor_df.empty:
        obj_minor = np.mean(smry_minor.loc[:,"Y"].values)
    else:
        obj_minor = np.nan
    
    if not rest_df.empty:
        obj_rest = np.mean(smry_rest.loc[:,"Y"].values)
    else:
        obj_rest = np.nan

    print(A_name, "obj     ", 
          "all:",round(obj_all,3), 
          "minor:",round(obj_minor,3),
          "rest:",round(obj_rest,3))

    return obj_all, obj_minor, obj_rest

