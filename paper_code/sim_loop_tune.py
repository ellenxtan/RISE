"""
RISE: Robust Individualized decision learning with SEnsitive variables

Author: Xiaoqing (Ellen) Tan 
"""

from imports import *
from myfuns_tune import *

# Updated 2022-07-26: call R package
# conda install -c r rpy2
# which R  
# R  #then install R packages under that R path
os.environ['R_HOME'] = '/Users/xtan/miniforge3/lib/R'
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects


def sim_loop(proj, i_sim, df_all, s_type, X_names, XS_names, names_to_norm, qua_use, 
             is_sim, y_fn, is_class, is_qua, is_inf, is_rct,
             is_tune, param_grid, mypath, is_save):

    print("*"*15,proj,"i_sim =",i_sim,"begin","*"*15)
    seed = i_sim + 12345
    set_random(seed)

    ########################################################################################################################
    ###################################################### Preprocess ######################################################
    ########################################################################################################################

    # train-test split
    df_tr_ori, df_te_ori = get_train_test(df_all, s_type, seed_value=i_sim)
    print("df_tr_ori.head(3)\n", df_tr_ori.head(3))

    # normalize
    if not is_sim: # only normalize for real (sim already in [0,1])
        df, df_test = normalize(df_tr_ori, df_te_ori, names_to_norm)
    else:
        df      = df_tr_ori.copy()
        df_test = df_te_ori.copy()
    print("df.head(3)\n", df.head(3))

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

    ########################################################################################################################
    #################################################### Begin Analysis ####################################################
    ########################################################################################################################

    print("begin analysis")

    df0 = df[df["A"]==0].reset_index(drop=True).copy()
    df1 = df[df["A"]==1].reset_index(drop=True).copy()

    df0_te = df_test[df_test["A"]==0].reset_index(drop=True).copy()
    df1_te = df_test[df_test["A"]==1].reset_index(drop=True).copy()

    # Baseline E(Y|X,A) (no S)
    is_base = True
    obj_name = "_base_"
    g_base0, _ = fit_expectation(obj_name, df0, X_names, df, is_class, is_tune, param_grid, df0_te)
    g_base1, _ = fit_expectation(obj_name, df1, X_names, df, is_class, is_tune, param_grid, df1_te)

    y_base, w_base = get_label_wt(g_base1, g_base0)
    A_base, A_base_tr, model_base = class_pred(df, df_test, X_names, y_base, w_base, s_type, is_tune, param_grid)

    # EXP E(Y|X,S,A) # XS train, X predict
    is_exp = True
    obj_name = "_exp_"  
    g_exp0, _ = fit_expectation(obj_name, df0, XS_names, df, is_class, is_tune, param_grid, df0_te)
    g_exp1, _ = fit_expectation(obj_name, df1, XS_names, df, is_class, is_tune, param_grid, df1_te)

    y_exp, w_exp = get_label_wt(g_exp1, g_exp0)
    A_exp, A_exp_tr, model_exp = class_pred(df, df_test, X_names, y_exp, w_exp, s_type, is_tune, param_grid)

    # QUA/INF E(Y|X,S,A)
    if (is_qua or is_inf):
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
        A_qua, A_qua_tr, model_qua = class_pred(df, df_test, X_names, y_qua, w_qua, s_type, is_tune, param_grid)

    # Updated 2022-07-26: policytree (PT-Base and PT-EXP)
    # Calling R
    pandas2ri.activate()
    numpy2ri.activate()
    grf = importr("grf")
    policytree = importr("policytree")

    obj_name = "_pt_base_"
    c_forest = grf.causal_forest(df[X_names], df["Y"], df["A"])
    dr_scores = policytree.double_robust_scores(c_forest)
    model_pt_base = policytree.policy_tree(df[X_names], dr_scores)
    A_ptbase = robjects.r.predict(model_pt_base, df_test[X_names])
    A_ptbase_tr = robjects.r.predict(model_pt_base, df[X_names]) #1 corresponds to control, and 2 treated
    A_ptbase = np.where(A_ptbase==2., "yes", "no")
    A_ptbase_tr = np.where(A_ptbase_tr==2., "yes", "no")
    print(obj_name, "pred_train table:", collections.Counter(A_ptbase_tr.ravel()))
    print(obj_name, "pred_test table:", collections.Counter(A_ptbase.ravel()))
    print("="*30)

    obj_name = "_pt_exp_"
    c_forest = grf.causal_forest(df[XS_names], df["Y"], df["A"])
    dr_scores = policytree.double_robust_scores(c_forest)
    model_pt_exp = policytree.policy_tree(df[X_names], dr_scores) #only X is used in final model
    A_ptexp = robjects.r.predict(model_pt_exp, df_test[X_names])
    A_ptexp_tr = robjects.r.predict(model_pt_exp, df[X_names])
    A_ptexp = np.where(A_ptexp==2., "yes", "no")
    A_ptexp_tr = np.where(A_ptexp_tr==2., "yes", "no")
    print(obj_name, "pred_train table:", collections.Counter(A_ptexp_tr.ravel()))
    print(obj_name, "pred_test table:", collections.Counter(A_ptexp.ravel()))
    print("="*30)

    ########################################################################################################################
    ################################################### Evaluate Metrics ###################################################
    ########################################################################################################################

    # add A_pred to df
    A_names = get_A_names(is_base, is_exp, is_qua, is_inf)
    res_df_te = add_Apred_Midx(df_test, A_names, A_base, A_exp, A_qua)

    # Updated 2022-07-26: policytree (PT-Base and PT-EXP)
    A_names = A_names + ["A_ptbase","A_ptexp"]
    res_df_te["A_ptbase"] = A_ptbase
    res_df_te["A_ptexp"] = A_ptexp

    # get optimal A,S given X (QUA/INF optimal)
    minor_name = 'M_opt'
    #test
    res_df_te['S_opt'], res_df_te[minor_name], df_S_opt_te = \
                optimal_AS_inf(df_test, proj, is_sim, s_type, X_names, XS_names, y_fn, is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1)
    if not df_S_opt_te.empty:
        minor_rule(df_S_opt_te, X_names, s_type, "S_opt")

    # metrics
    res_te = eval_metrics(minor_name, res_df_te, A_names, X_names, XS_names, y_fn, s_type, is_class, qua_use, fit_Y_XS_A0, fit_Y_XS_A1, fit_qua0, fit_qua1)
    
    # save
    if is_save and i_sim==0:
        # save pred A
        res_df_te.to_csv(mypath+proj+"_"+s_type+"_res_df_te_"+str(i_sim)+".csv", index=False)

    print("*"*15,proj,"i_sim=",i_sim,"finish","*"*15)

    return res_te
