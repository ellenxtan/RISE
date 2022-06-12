from .utils import *
import sys

class output:
    def __init__(self, A_pred, is_vul, obj_all, val_all, obj_vul, val_vul, model_rise):
        self.A_pred = A_pred
        self.is_vul = is_vul
        self.obj_all = obj_all
        self.val_all = val_all
        self.obj_vul = obj_vul
        self.val_vul = val_vul
        self.model = model_rise


class out_model:
    def __init__(self, seed, df, df_test, X_names, y_qua, w_qua, s_type, is_tune, param_grid):
        self.seed = seed
        self.df = df
        self.df_test = df_test
        self.X_names = X_names
        self.y_qua = y_qua
        self.w_qua = w_qua
        self.s_type = s_type
        self.is_tune = is_tune
        self.param_grid = param_grid


def rise(train_df, test_df, Y_name, A_name, X_names, S_names, is_rct, is_class, s_type, 
         qua_use=None, is_tune=False, param_grid=None, is_plot=True, seed=12345):

    # sanity check
    try: 
        assert s_type in ["cont", "disc", "multi-disc"]
    except:
        sys.exit('s_type should be in ["cont", "disc", "multi-disc"]')

    if s_type in ["disc", "multi-disc"]:
        qua_use = None
    
    if is_tune:
        try:
            assert param_grid is not None
        except:
            sys.exit('when is_tune=True, param_grid should not be None')
    else:
        param_grid = None

    # default values
    is_sim = False
    proj, y_fn = None, None

    # reorganize data
    train_df.rename(columns={Y_name:"Y"}, inplace=True)
    test_df.rename(columns={Y_name:"Y"}, inplace=True)

    train_df.rename(columns={A_name:"A"}, inplace=True)
    test_df.rename(columns={A_name:"A"}, inplace=True)
    if set(train_df["A"]) != set([0,1]): # randomly pick one as control/treatment
        assert set(train_df["A"]) == set(test_df["A"])
        a0 = list(set(train_df["A"]))[0]
        train_df["A"] = np.where(train_df["A"]==a0, 0, 1)
        test_df["A"] = np.where(test_df["A"]==a0, 0, 1)

    XS_names = S_names + X_names

    # discrete var has <=5 values #TODO: may need to change in future
    if s_type == "cont":
        names_to_norm = list(train_df[XS_names].nunique()[train_df[XS_names].nunique() > 5].index)
    else: # not normalize S if disc/multi-disc
        names_to_norm = list(train_df[X_names].nunique()[train_df[X_names].nunique() > 5].index)

    set_random(seed)


    ############################################################################
    ################################ Preprocess ################################
    ############################################################################
    
    # normalize
    df, df_test = normalize(train_df, test_df, names_to_norm)

    # PS based on train
    if is_rct:
        pr_a1 = sum(df["A"]==1) / df.shape[0]
        fit_ps = None
        df_test["ps"] = pr_a1
        df["ps"] = pr_a1
        print("pr_a1", pr_a1)
    else:
        pr_a1 = None
        fit_ps = get_ps_fit(df, XS_names) # propensity score A|XS
        df_test["ps"] = fit_ps.predict_proba(df_test[XS_names])[:,1]
        df["ps"] = fit_ps.predict_proba(df[XS_names])[:,1]
    
    df_test["ps"] = np.where(df_test["A"]==1, df_test["ps"], 1-df_test["ps"])
    df["ps"] = np.where(df["A"]==1, df["ps"], 1-df["ps"])
    
    # PS trim using percentile cutpoints (only for observational study)
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3069059/
    if not is_rct:
        print("for observational study, trimming PS using percentile cutpoints")
        
        ps_q90 = df["ps"].quantile(0.9)
        ps_q10 = df["ps"].quantile(0.1)
        print("train: before ps min:", np.min(df["ps"]), "ps max:", np.max(df["ps"]))
        df["ps"] = np.where(df["ps"] > ps_q90, ps_q90, df["ps"])
        df["ps"] = np.where(df["ps"] < ps_q10, ps_q10, df["ps"])
        print("train: after  ps min:", np.min(df["ps"]), "ps max:", np.max(df["ps"]))

        ps_q90 = df_test["ps"].quantile(0.9)
        ps_q10 = df_test["ps"].quantile(0.1)
        print("test: before ps min:", np.min(df_test["ps"]), "ps max:", np.max(df_test["ps"]))
        df_test["ps"] = np.where(df_test["ps"] > ps_q90, ps_q90, df_test["ps"])
        df_test["ps"] = np.where(df_test["ps"] < ps_q10, ps_q10, df_test["ps"])
        print("test: after  ps min:", np.min(df_test["ps"]), "ps max:", np.max(df_test["ps"]))


    ############################################################################
    ############################## Begin Analysis ##############################
    ############################################################################

    df0 = df[df["A"]==0].reset_index(drop=True).copy()
    df1 = df[df["A"]==1].reset_index(drop=True).copy()

    df0_te = df_test[df_test["A"]==0].reset_index(drop=True).copy()
    df1_te = df_test[df_test["A"]==1].reset_index(drop=True).copy()

    # QUA/INF E(Y|X,S,A)
    set_random(seed)
    df0["Yhat"], fit_Y_XS_A0 = fit_expectation("E(Y|X,S,A=0)", df0, XS_names, df0, is_class, is_tune, param_grid, df0_te)
    df1["Yhat"], fit_Y_XS_A1 = fit_expectation("E(Y|X,S,A=1)", df1, XS_names, df1, is_class, is_tune, param_grid, df1_te)

    fit_qua0, fit_qua1 = None, None
    if (s_type == "cont"): # quantile: Yhat|X,A (no S)
        set_random(seed)
        fit_qua0, g_qua0 = fit_quantile(df0[X_names], df0["Yhat"], df[X_names], qua_use, is_tune, param_grid)
        fit_qua1, g_qua1 = fit_quantile(df1[X_names], df1["Yhat"], df[X_names], qua_use, is_tune, param_grid)

    if (s_type == "disc"): # infinite: try all s and find s that helps achieve a minimal
        g_qua0 = fit_infinite(df, X_names, XS_names, fit_Y_XS_A0)
        g_qua1 = fit_infinite(df, X_names, XS_names, fit_Y_XS_A1)

    y_qua, w_qua = get_label_wt(g_qua1, g_qua0)
    set_random(seed)
    A_te, A_tr, model_rise = class_pred(df, df_test, X_names, y_qua, w_qua, s_type, is_tune, param_grid)

    if is_plot:
        out_mod = out_model(seed, df, df_test, X_names, y_qua, w_qua, s_type, is_tune, param_grid)
    else:
        out_mod = None


    ############################################################################
    ############################# Evaluate Metrics #############################
    ############################################################################

    # add A_pred to df
    A_name = 'A_rise'
    M_name = 'M_opt'
    df_test[A_name] = A_te

    # get optimal S given X (QUA/INF optimal)
    if s_type in ["cont", "disc"]:
        df_test[M_name] = optimal_AS_inf(df_test, proj, is_sim, s_type, X_names, 
                    XS_names, y_fn, is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1)
    else: # "multi-disc"
        df_test[M_name] = optimal_AS_inf_mulS(df_test, proj, is_sim, s_type, X_names, 
                    XS_names, y_fn, is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1)
    
    # metrics
    if is_rct:
        val_all, val_vul, value_rest = expect_reward(df_test, A_name, M_name)
    else:
        val_all, val_vul, value_rest = get_value_fn(df_test, A_name, XS_names, M_name, y_fn, fit_Y_XS_A0, fit_Y_XS_A1)    
    
    if s_type == "disc" or y_fn is not None: # real with disc S | sim with disc/cont S
        obj_all, obj_vul, obj_rest = get_obj_inf(df_test, X_names, XS_names, A_name, M_name, y_fn, s_type, is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1)
    elif s_type == "cont": # real with cont S
        obj_all, obj_vul, obj_rest = get_obj_qua(df_test, X_names, s_type, A_name, M_name, fit_qua0, fit_qua1)

    # out
    A_pred = df_test[A_name].values
    is_vul = df_test[M_name].values
    out_res = output(A_pred, is_vul, obj_all, val_all, obj_vul, val_vul, model_rise)

    return out_res, out_mod
