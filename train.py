import pickle

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

COLUMNS = ['v2', 'v3', 'v8', 'v9', 'v10', 'v11', 'v12', 'v14', 'v15', 'v17', 'v19',
       'v1_a', 'v1_b', 'v4_l', 'v4_u', 'v4_y', 'v5_g', 'v5_gg',
       'v5_p', 'v6_W', 'v6_aa', 'v6_c', 'v6_cc', 'v6_d', 'v6_e', 'v6_ff',
       'v6_i', 'v6_j', 'v6_k', 'v6_m', 'v6_q', 'v6_r', 'v6_x', 'v7_bb',
       'v7_dd', 'v7_ff', 'v7_h', 'v7_j', 'v7_n', 'v7_o', 'v7_v', 'v7_z',
       'v13_g', 'v13_p', 'v13_s', 'v18_f', 'v18_t']

# For data preparation and validation. 
# This can be inferred programmatically as well!
VARIABLES = {
    "not_null": {'v3', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v15', 'v19'},
    "bool_str": {'v9', 'v10', 'v12'},
    "dummy": {'v1', 'v4', 'v5', 'v6', 'v7', 'v13', 'v18'},
    "cont_nullable": {'v2', 'v14', 'v17'}
}

def load_data():
    df = pd.read_csv('./data/training.csv', sep=';', decimal=",")
    df_valid = pd.read_csv('./data/validation.csv', sep=';', decimal=",")
    df = pd.concat([df, df_valid])

    # not null and binary strings can be a single 1/0 value
    for col in VARIABLES['bool_str']:
        df[col] = (df[col]=='t').astype(int)

    for col in VARIABLES['dummy']:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop([col], axis=1), dummies], axis=1)
    return df

def cv(df):
    X = df[COLUMNS].fillna(0).values
    y = (df['classLabel'] == 'yes.').astype(int).values

    estimator = XGBClassifier()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(estimator, X, y, cv=kfold, scoring='f1')
    print('CV Results: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))

def train(df):
    X = df[COLUMNS].fillna(0).values
    y = (df['classLabel'] == 'yes.').astype(int).values

    model = XGBClassifier()
    model.fit(X, y)
    pickle.dump(model, open('model.pkl', 'wb'))
    print('Trained model has been saved.')

def main():
    df = load_data()
    cv(df)
    train(df)

if __name__ == "__main__":
    main()

