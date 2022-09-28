"""
RISE: Robust Individualized decision learning with SEnsitive variables

Author: Xiaoqing (Ellen) Tan 
"""

from imports import *

def gen_sim_data(n, s_type, y_set):
    """generate simulated data"""
    
    # Example 1: toy example  # vulnerable: X1>0.5 -> S=0; X1<=0.5 -> S=1
    if (y_set == "toy"):
        y_fn = "(X1>0.5)*(5 + 10*A + 22*S - 24*A*S) + (X1<=0.5)*(11 + 19*A + 2*S - 32*A*S) + X2"
        X1 = np.random.uniform(0,1, n)
        X2 = np.random.uniform(0,0, n)
        err = np.random.normal(loc=0.0, scale=1, size=n)

        if s_type == "cont":
            s1 = np.random.beta(4,1,n//2)
            s0 = np.random.beta(1,4,n-n//2)
            S = np.concatenate([s0,s1])
        elif s_type == "disc":
            S = np.random.binomial(1, 0.5, n)
        A = np.random.binomial(1, 0.5, n)
        dat = np.concatenate((A[:,np.newaxis], S[:,np.newaxis], X1[:,np.newaxis], X2[:,np.newaxis]),axis=1) #, X3[:,np.newaxis], X4[:,np.newaxis]
        df = pd.DataFrame(data=dat, columns=['A','S','X1','X2'])
        df["Y"] = df.eval(y_fn) + err
    
    
    # Example 2 complex: observational + correlated XS + multiple X
    if (y_set == "complex"):
        y_fn = "(0.5 + 1*A + exp(S) - 2.5*A*S) * (1+X1 -X2 +X3**2 +exp(X4)) + (1 + 2*A + 0.2*exp(S) - 3.5*A*S) * (1+5*X1 -2*X2 +3*X3 +2*exp(X4))"
        X1 = np.random.uniform(0,1, n)
        X2 = np.random.uniform(0,1, n)
        X3 = np.random.uniform(0,1, n)
        X4 = np.random.uniform(0,1, n)
        X5 = np.random.uniform(0,1, n)
        X6 = np.random.uniform(0,1, n)
        err = np.random.normal(loc=0.0, scale=1, size=n)

        pr_s1 = 1 / (1 + np.exp( 2.5 - 0.8*(X1+X2+X3+X4+X5+X6) )) # expit(-2.5 + 0.8()) # norm

        if s_type == "cont":
            s1 = np.random.beta(4,1,n//2) #1 4/5=0.8
            s0 = np.random.beta(1,4,n-n//2) #0 1/5=0.2
            S = np.concatenate([s0,s1])
        else:
            S = np.random.binomial(1, pr_s1, n)
        
        dat = np.concatenate((S[:,np.newaxis], X1[:,np.newaxis], X2[:,np.newaxis], X3[:,np.newaxis], X4[:,np.newaxis], X5[:,np.newaxis], X6[:,np.newaxis]),axis=1)
        df = pd.DataFrame(data=dat, columns=['S','X1','X2','X3','X4','X5','X6'])

        # expit( 0.6(-S + X1 - X2 + X3 - X4 + X5 - X6) )
        a_fn = "1 / (1 + exp( 0.6*S - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))"
        prop_sc = df.eval(a_fn)
        df['A'] = np.random.binomial(1, prop_sc, n)

        df["Y"] = df.eval(y_fn) + err
    

    # S as a noise variable
    if (y_set == "noise"):
        y_fn = "(X1<=0.5)*(8+12*A+16*exp(X2)-26*A*X2) + (X1>0.5)*(13+3*A+2*exp(X2)-8*A*X2)"
        X1 = np.random.uniform(0,1, n) #(-4,4,n)
        X2 = np.random.uniform(0,1, n)
        err = np.random.normal(loc=0.0, scale=1, size=n)

        pr_s1 = 1 / (1 + np.exp( 2.5 - 2.5*(X1+X2) )) # =expit(-2.5+2.5(X1+X2))
        
        if s_type == "cont":
            S = pr_s1
        else:
            S = np.random.binomial(1, pr_s1, n)

        A = np.random.binomial(1, 0.5, n)
        dat = np.concatenate((A[:,np.newaxis], S[:,np.newaxis], X1[:,np.newaxis], X2[:,np.newaxis]),axis=1) #, X3[:,np.newaxis], X4[:,np.newaxis]
        df = pd.DataFrame(data=dat, columns=['A','S','X1','X2']) #,'X3','X4'
        df["Y"] = df.eval(y_fn) + err


    # Example 2 with positivity assump nearly violated
    if (y_set == "positivity"):
        y_fn = "(0.5 + 1*A + exp(S) - 2.5*A*S) * (1+X1 -X2 +X3**2 +exp(X4)) + (1 + 2*A + 0.2*exp(S) - 3.5*A*S) * (1+5*X1 -2*X2 +3*X3 +2*exp(X4))"
        X1 = np.random.uniform(0,1, n)
        X2 = np.random.uniform(0,1, n)
        X3 = np.random.uniform(0,1, n)
        X4 = np.random.uniform(0,1, n)
        X5 = np.random.uniform(0,1, n)
        X6 = np.random.uniform(0,1, n)
        err = np.random.normal(loc=0.0, scale=1, size=n)

        pr_s1 = 1 / (1 + np.exp( 2.5 - 0.8*(X1+X2+X3+X4+X5+X6) )) # expit(-2.5 + 0.8()) # norm

        if s_type == "cont":
            # S = pr_s1
            s1 = np.random.beta(4,1,n//2) #1 4/5=0.8
            s0 = np.random.beta(1,4,n-n//2) #0 1/5=0.2
            S = np.concatenate([s0,s1])
        else:
            S = np.random.binomial(1, pr_s1, n)
        
        dat = np.concatenate((S[:,np.newaxis], X1[:,np.newaxis], X2[:,np.newaxis], X3[:,np.newaxis], X4[:,np.newaxis], X5[:,np.newaxis], X6[:,np.newaxis]),axis=1)
        df = pd.DataFrame(data=dat, columns=['S','X1','X2','X3','X4','X5','X6'])

        # skew a_fn: expit( -1.2(-S + X1 - X2 + X3 - X4 + X5 - X6) )
        a_fn = "1 / (1 + exp( 1.2 * (-S + X1 - X2 + X3 - X4 + X5 - X6) ))" 

        prop_sc = df.eval(a_fn)

        df['A'] = np.random.binomial(1, prop_sc, n)

        df["Y"] = df.eval(y_fn) + err
    

    # Example 2 with unmeasured confouding
    if (y_set == "unconfound"):
        y_fn = "(0.5 + 1*A + exp(S) - 2.5*A*S) * (1+X1 -X2 +X3**2 +exp(X4)) + (1 + 2*A + 0.2*exp(S) - 3.5*A*S) * (1+5*X1 -2*X2 +3*X3 +2*exp(X4))"
        X1 = np.random.uniform(0,1, n)
        X2 = np.random.uniform(0,1, n)
        X3 = np.random.uniform(0,1, n)
        X4 = np.random.uniform(0,1, n)
        X5 = np.random.uniform(0,1, n)
        X6 = np.random.uniform(0,1, n)
        err = np.random.normal(loc=0.0, scale=1, size=n)

        if s_type == "cont":
            # S = pr_s1
            s1 = np.random.beta(4,1,n//2) #1 4/5=0.8
            s0 = np.random.beta(1,4,n-n//2) #0 1/5=0.2
            S = np.concatenate([s0,s1])
        else:
            pr_s1 = 1 / (1 + np.exp( 2.5 - 0.8*(X1+X2+X3+X4+X5+X6) )) # expit(-2.5 + 0.8()) # norm
            S = np.random.binomial(1, pr_s1, n)
        
        dat = np.concatenate((S[:,np.newaxis], X1[:,np.newaxis], X2[:,np.newaxis], X3[:,np.newaxis], X4[:,np.newaxis], X5[:,np.newaxis], X6[:,np.newaxis]),axis=1)
        df = pd.DataFrame(data=dat, columns=['S','X1','X2','X3','X4','X5','X6'])

        a_fn = "1 / (1 + exp( 0.6*S - 0.6*X1 + 0.6*X2 - 0.6*X3 + 0.6*X4 - 0.6*X5 + 0.6*X6 ))" # expit( -0.6(S + X1 - X2 + X3 - X4 + X5 - X6) )
        prop_sc = df.eval(a_fn)
        df['A'] = np.random.binomial(1, prop_sc, n)

        df["Y"] = df.eval(y_fn) + err

        df["X1_err"] = df["X1"] + np.random.normal(loc=0.0, scale=1, size=n)
        df["X1_err"] = (df["X1_err"] - np.min(df["X1_err"])) / (np.max(df["X1_err"]) - np.min(df["X1_err"])) # scale to [0,1]
        print("df['X1'].min, df['X1'].max", np.min(df['X1']), np.max(df['X1']))
        print("df['X1_err'].min, df['X1_err'].max", np.min(df['X1_err']), np.max(df['X1_err']))


    # convert to float
    cols=[i for i in df.columns]
    for col in cols:
        df[col]=pd.to_numeric(df[col])

    df["A"] = df["A"].astype(int)

    if (s_type == "disc"):
        df["S"] = df["S"].astype(int)

    return(df, y_fn)

