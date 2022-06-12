To run RISE

```py
from rise import utils as utils
from rise import rise as r
import pandas as pd

train_df = pd.read_csv("data/cont_df_train.csv")
test_df = pd.read_csv("data/cont_df_test.csv")

Y_name = "Y"
A_name = "A"
X_names = ["X1","X2","X3","X4","X5","X6"]
S_names = ["S"]
is_rct = False
is_class = False
s_type = "cont"
qua_use = 0.25

out, out_mod = r.rise(train_df, test_df, Y_name, A_name, X_names, S_names, 
                      is_rct, is_class, s_type, qua_use, 
                      is_tune=False, is_plot=True, seed=12345)
```


Outputs

```py
out.A_pred
out.is_vul
out.obj_all
out.val_all
out.obj_vul
out.val_vul
out.model
```

SHAP plot

```py
from rise import plots as p

p.plot_shap(out_mod, save_path="./")
```