import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder


def conv_ric_to_tic():
    """Converts Refinitive Security identification scheme to conventional Stock Market symbols"""
    path = r'assets/fundamentals'  # use your path
    all_files = glob.glob(path + "/*.csv")
    for f in all_files:
        df = pd.read_csv(f)
        inst_ric = df['Instrument']
        tics = [i.split('.')[0] for i in inst_ric]
        tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
        tics = ['BF-B' if i == 'BFb' else i for i in tics]
        df['Instrument'] = tics
        df.to_csv(f)
    return


def create_price_ratios():
    """Creates price valuation metrics from raw data"""
    mkt_cap = pd.read_csv("assets/fundamentals/mktcap_df.csv", index_col=0)
    net_inc = pd.read_csv("assets/fundamentals/Net_income_df.csv", index_col=0)
    book = pd.read_csv("assets/fundamentals/Book_df.csv", index_col=0)
    mkt_cap_1 = pd.merge(mkt_cap, book, on=["Original Announcement Date Time", "Instrument"])
    mkt_cap = pd.merge(mkt_cap_1, net_inc, on=["Original Announcement Date Time", "Instrument"]).dropna()
    mkt_cap["e_p"] = mkt_cap["Income Available to Common Shares"] / mkt_cap["Market Capitalization"]
    mkt_cap["b_p"] = mkt_cap['Common Equity - Total'] / mkt_cap["Market Capitalization"]
    price_features = mkt_cap.drop(['Common Equity - Total', 'Income Available to Common Shares',
                                   'Market Capitalization'], axis=1)
    price_features.to_csv('assets/fundamentals/price_ratios.csv', index=False)

    return


def create_ev_ratios():
    """creates enterprise value valuation metrics from raw data"""
    ev = pd.read_csv("assets/fundamentals/EV_df.csv", index_col=0)
    ebit = pd.read_csv("assets/fundamentals/EBIT_df.csv", index_col=0)
    rev = pd.read_csv("assets/fundamentals/Revenue_df.csv", index_col=0)
    ev_1 = pd.merge(ev, ebit, on=["Original Announcement Date Time", "Instrument"])
    ev = pd.merge(ev_1, rev, on=["Original Announcement Date Time", "Instrument"]).dropna()
    ev["ebit_ev"] = ev["Earnings before Interest & Taxes (EBIT) - Normalized"] / ev["Enterprise Value"]
    ev["rev_ev"] = ev["Revenue from Business Activities - Total"] / ev["Enterprise Value"]
    ev_features = ev.drop(["Enterprise Value", "Revenue from Business Activities - Total",
                           "Earnings before Interest & Taxes (EBIT) - Normalized"], axis=1).drop_duplicates()

    ev_features.to_csv('assets/fundamentals/ev_ratios.csv', index=False)

    return


def combine_ratios():
    """combines enterprise value and price valuation multiples."""
    price = pd.read_csv("assets/fundamentals/price_ratios.csv")
    ev = pd.read_csv("assets/fundamentals/ev_ratios.csv")
    combined = pd.merge(price, ev, on=["Original Announcement Date Time", "Instrument"]).drop_duplicates()
    combined.to_csv("assets/fundamentals/valuation_features.csv", index=False)

    return


def mom_calc():
    """Creates price momentum features for 6 and 12 month periods"""
    px_df = pd.read_csv("assets/fundamentals/price_dat.csv")
    vx_df = pd.read_csv("assets/fundamentals/valuation_features.csv")
    dates = list(vx_df['Original Announcement Date Time'])
    adj_dates = [i.split("T")[0] for i in dates]
    vx_df['Original Announcement Date Time'] = adj_dates
    vx_df = vx_df.rename(columns={"Original Announcement Date Time": "Date"})
    px_df = px_df.set_index('Date')
    val_mo_df = pd.DataFrame()
    production_df = pd.DataFrame()

    for stock in vx_df['Instrument'].unique():
        px_dat = px_df.filter([stock], axis = 1)
        vx_dat = vx_df[vx_df['Instrument']==stock].set_index('Date')
        vx_dat = vx_dat.join(px_dat).sort_index()
        vx_dat = vx_dat.rename(columns={stock: 'Price'})
        vx_dat['90 days'] = vx_dat['Price'].pct_change()
        vx_dat['1_yr'] = vx_dat['Price'].pct_change(4)
        vx_dat = vx_dat.dropna()
        vx_dat['fwd_ret'] = vx_dat['90 days'].shift(periods = -8, fill_value = 'xxx')
        prod = vx_dat[vx_dat['fwd_ret']=='xxx']
        valmo = vx_dat[vx_dat['fwd_ret']!='xxx']
        production_df = pd.concat([production_df, prod])
        val_mo_df = pd.concat([val_mo_df, valmo])
    val_mo_df = val_mo_df.drop('Price', axis = 1)
    production_df = production_df.drop('Price', axis= 1)
    val_mo_df.to_csv("assets/fundamentals/val_mo_features.csv")
    production_df.to_csv("assets/fundamentals/production.csv")
    return



def data_prep(path, set):
    """Creates the y vector, and one hot encoding of sectors"""
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"])
    qrt = list(data["Date"].dt.quarter)
    yr =  list(data["Date"].dt.year)
    q_dat = ['{}-{}'.format(qrt[i], yr[i]) for i in range(0, len(yr))]
    data['Timeframe'] = q_dat
    dat_df = pd.DataFrame()
    for d in data['Timeframe'].unique():
        x = data[data['Timeframe']==d]
        try:
            x['quartiles'] = pd.qcut(x["fwd_ret"], 2, labels=False, duplicates='drop')
            dat_df = pd.concat([dat_df, x])
        except:
            x['quartiles'] = ['xxx' for i in range(0, len(x))]
            dat_df = pd.concat([dat_df, x])
    sectors = pd.read_csv("assets/fundamentals/sectors.csv", index_col=0)
    data = dat_df
    data = data.merge(sectors, on="Instrument", how='outer')
    ohe = OneHotEncoder(sparse=False)
    ohe_dat = ohe.fit_transform(data["GICS Sector"].to_numpy().reshape(-1, 1))
    ohe_df = pd.DataFrame(ohe_dat, columns=ohe.get_feature_names())
    data = pd.concat([data, ohe_df], axis=1).sort_values("Date")
    data = data.dropna()
    data.to_csv("assets/fundamentals/clean_raw_{}.csv".format(set), index=False)
    return

def feature_prep(path):
    """Preps features for ML classification model"""
    raw = pd.read_csv(path)
    X_y_df = raw.drop(['Date', 'Instrument', 'fwd_ret', "Timeframe", "GICS Sector"], axis=1)
    X = X_y_df.drop('quartiles', axis=1).to_numpy()
    y = X_y_df['quartiles'].to_numpy()

    return X, y

def pipeline(X, y):
    """Trains, tests, and evaluates classification model using GridSearch"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    param_grid = {'n_estimators': [200, 500],
                  'max_depth': [4, 5, 6, 7],
                  "max_features": ['auto', 'sqrt', 'log2'],
                  "criterion": ['gini', 'entropy'],
                  "bootstrap": [False, True]}
    scorers = {
        'precision_score': make_scorer(precision_score),
        'fbeta_score': make_scorer(fbeta_score, beta = 1.5),
        'recall_score':make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)}

    etc = ExtraTreesClassifier(random_state=42)
    clf = GridSearchCV(etc, param_grid=param_grid, cv=5, refit='fbeta_score',
                       return_train_score=True, scoring=scorers)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    t = clf.best_params_
    print(t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    results = pd.DataFrame(clf.cv_results_)
    results.to_csv('mod_1_gsresults.csv')
    return t, X_train, X_test, y_train, y_test, clf

def dec_thresh_adj(X_test,y_test, clf):
    """adjusts decision threshold if necessary"""
    y_scores = clf.predict_proba(X_test)[:, 1]
    thresh_ranges = np.linspace(0.5, 0.7, 41)
    prec_info = []
    for t in thresh_ranges:
        adj_score = [1 if y > t else 0 for y in y_scores]
        prec = fbeta_score(y_test, adj_score, beta=1.5)
        prec_info.append((t, prec))
        print(pd.DataFrame(confusion_matrix(y_test, adj_score),
                           columns=['pred_neg', 'pred_pos'],
                           index=['neg', 'pos']))
    decision_vals  = max(prec_info, key=lambda i: i[1])
    dec_factor = decision_vals[0]
    prec_val = decision_vals[1]
    print("Decision Threshold Adjusted to {} new fbeta score {}".format(dec_factor, prec_val))
    return dec_factor

def mod_1_output(clf, t):
    """Applies selected model to production data"""
    X_prod, y_prod = feature_prep("assets/fundamentals/clean_raw_production.csv")
    y_pred = clf.predict_proba(X_prod)[:, 1]
    adj_score = [ "BUY" if y > t else "SELL" for y in y_pred]
    df = pd.read_csv("assets/fundamentals/clean_raw_production.csv")
    df['BUY_SELL'] = adj_score
    df['Probability'] = y_pred
    df = df.filter(["Date","Instrument", "GICS Sector", "BUY_SELL", "Probability"])
    df.to_csv('assets/fundamentals/Production_output.csv')
    return


if __name__ == "__main__":
    create_price_ratios()
    create_ev_ratios()
    combine_ratios()
    mom_calc()
    data_prep("assets/fundamentals/val_mo_features.csv", 'train_test')
    data_prep("assets/fundamentals/production.csv", 'production')
    X, y = feature_prep("assets/fundamentals/clean_raw_train_test.csv")
    t, X_train, X_test, y_train, y_test, clf = pipeline(X, y)
    thresh = dec_thresh_adj(X_test,y_test, clf)
    mod_1_output(clf, thresh)

