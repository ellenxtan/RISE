from .utils import *
import shap
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

# save SHAP plots
def plot_shap(out, save_path): #df_test[X_names]
    tf.compat.v1.disable_v2_behavior()
    _, _, model = class_pred(out.df, out.df_test, out.X_names, out.y_qua, out.w_qua, out.is_tune, out.s_type)

    explainer = shap.DeepExplainer(model, out.df[out.X_names])
    shap_values = explainer.shap_values(out.df_test[out.X_names].values)
    # shap.summary_plot(shap_values[0], plot_type = 'bar', feature_names = out.df_test[out.X_names].columns)
    shap_abs(shap_values[0], out.df_test[out.X_names], save_path+"fig_rise")


def shap_abs(df_shap, df_in, fig_name):
    """
    Simplified version of SHAP representation for easier interpretation
    modified based on https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
    """
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df_in.columns
    shap_v.columns = feature_list
    df_v = df_in.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_name+'.pdf')


