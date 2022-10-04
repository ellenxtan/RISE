"""
RISE: Robust Individualized decision learning with SEnsitive variables

Author: Xiaoqing (Ellen) Tan 

Simulation settings
"toy": Example 1
"complex": Example 2
"noise": S as a noise variable
"positivity": Example 2 with near violation of positivity assumption
"unconfound": Example 2 with violation of unconfoundedness assumption
"""

from imports import *
from loop import *
from dat import *

# settings
proj    = "simulation"
y_set   = "toy"  #"complex" #"noise" #"positivity" #"unconfound" #
s_type  = "cont" #"disc" #
qua_use = 0.25

n_train = 10000 #train-test: 8-2
n_sim   = 100
is_tune = False #True #
is_save = False
mypath  = ""

if (proj == "simulation"):
    df_all_ori, y_fn = gen_sim_data(n_train, s_type, y_set)
    is_sim = True
else:
    is_sim = False
    y_fn = None
    y_set = None

if proj in ["simulation"]:
    is_class = False

if (s_type == "disc"):
    is_qua = False
    is_inf = True
else: #cont
    is_qua = True
    is_inf = False

if y_set in ['toy','noise']:
    is_rct = True
elif y_set in ['complex','positivity','unconfound']:
    is_rct = False
else:
    assert(1==0)

if (is_tune):
    param_grid = dict(layers=[1,2,3],              #1,2,3
                      nodes=[256,512,1024],            #256,512,1024
                      dropouts=[0.1,0.2,0.3],          #0.1,0.2,0.3
                      acts=["sigmoid","relu","tanh"], #"sigmoid","relu","tanh"
                      opts=["adam","nadam","adadelta"],           #"adam","nadam","adadelta"
                      bsizes=[32,64,128],             #32,64,128
                      n_epochs=[50,100,200]            #50,100,200
                     )
else:
    param_grid = None

if not os.path.exists('logs'):
    os.makedirs('logs')

# names of all X
df_all = df_all_ori.copy()
if is_sim:
    X_names = list(df_all.columns[np.flatnonzero(np.char.startswith(list(df_all.columns), 'X'))])
    S_names = ["S"]
else:
    X_names = list( set(df_all.columns) - set(["A","Y","S"]) )
    S_names = ["S"]
XS_names = S_names + X_names

if not is_sim:
    names_to_norm = list(df_all[XS_names].nunique()[df_all[XS_names].nunique() > 2].index) #TODO:discrete S has >5 values
else:
    names_to_norm = []
print("X_names:", X_names)
print("S_names:", S_names)
print("cont vars (need to normalize):", names_to_norm)
print("disc vars (no need to normalize):", list(set(XS_names) - set(names_to_norm)))

# begin running
i_sim = 0
dict_list_est = list()
start0 = time.time()
start_time = time.strftime("%H:%M:%S", time.localtime())
for i_sim in range(n_sim):
    start = time.time()

    if is_sim:
        set_random(i_sim+12345)
        df_all_ori, y_fn = gen_sim_data(n_train, s_type, y_set)
        df_all = df_all_ori.copy()
        
    res_est = sim_loop(proj, i_sim, df_all, s_type, X_names, XS_names, names_to_norm, qua_use, 
             is_sim, y_fn, is_class, is_qua, is_inf, is_rct,
             is_tune, param_grid, mypath, is_save)
    
    dict_list_est.append(res_est)

    if i_sim % 1 == 0:
        _, tmp_df = dict_mean(dict_list_est)
        _, tmp_std_df = dict_std(dict_list_est)
        print(i_sim, "M_test mean\n", tmp_df.round(3))
        print(i_sim, "M_test std\n",  tmp_std_df.round(3))
        print("*"*60)

    end = time.time()
    print("iter", i_sim, "takes", (end - start)//60, "mins")
    print("start time", start_time)
    print("current process takes", (end - start0)//60, "mins")
    print("*"*80)


# final result smry
mean_dict_est, mean_df_est = dict_mean(dict_list_est)
std_dict_est, std_df_est = dict_std(dict_list_est)
print("*"*60)
print(mean_df_est.round(3))
print(std_df_est.round(3))

end0 = time.time()
print("whole process takes", (end0 - start0)//60, "mins")
print("*"*80)

# save result
if is_save:
    smry_res_all = [dict_list_est, mean_dict_est, mean_df_est, std_dict_est, std_df_est]
    out_file = mypath+proj+"_"+s_type+'_smry_res'
    mean_df_est.to_csv(out_file+"_est_mean.csv")
    std_df_est.to_csv(out_file+"_est_std.csv")
    with open(out_file+'.pickle', 'wb') as f:
        pickle.dump(smry_res_all, f)
