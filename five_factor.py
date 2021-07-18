import pickle
from yahoo_fin import stock_info as si
import re
from datetime import datetime
from threading import Event
import eikon as ek
import pandas as pd
from eikon.tools import get_date_from_today
import config
import DatastreamDSWS as DSWS

ek.set_app_key(config.ek_key())
ds = DSWS.Datastream(username=config.username_ds(), password=config.pw_ds())

def batch(iterable, n=1):
    """Helper function that assist in managing batch sizes"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def combine_inc_bal():
    """"""
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    df_income = pd.read_csv("assets/models/jeff_multi_factor/income_stat_dat.csv", index_col=0)
    df_balance = pd.read_csv("assets/models/jeff_multi_factor/bal_sht_dat.csv", index_col=0)
    df_cf = pd.read_csv("assets/models/jeff_multi_factor/cf_dat.csv", index_col=0)
    inc_bal = pd.DataFrame()
    for r in rics:
        inc = df_income[df_income['Instrument'] == r]
        bal = df_balance[df_balance['Instrument'] == r]
        cf = df_cf[df_cf['Instrument'] == r]
        inc_bal_1 = inc.merge(bal, on=['Instrument', "Original Announcement Date Time"])
        inc_bal_1 = inc_bal_1.merge(cf, on=['Instrument', "Original Announcement Date Time"])
        inc_bal_1 = inc_bal_1.rename(columns={'Original Announcement Date Time': 'date',
                                              'EPS - Basic - incl Extraordinary Items, Common - Total': 'eps_basic',
                                              'EPS - Basic - excl Extraordinary Items, Common - Total':'eps_excl_basic',
                                              'EPS - Diluted - excl Exord Items Applicable to Common Total': 'eps_dil',
                                              'EPS Basic incl Exord, Common - Total, PoP Diff': 'eps_growth',
                                              'Earnings before Interest & Taxes (EBIT)': 'ebit',
                                              'Earnings before Interest Taxes Depreciation & Amortization': 'ebitda',
                                              'Income Available to Common Shares': 'net_inc',
                                              'Revenue from Business Activities - Total': 'rev',
                                              'Total Assets': 'tot_assets',
                                              'Other Assets - Total': 'oth_assets',
                                              'Cash & Short Term Investments - Total': 'cash',
                                              "Total Shareholders' Equity incl Minority Intr & Hybrid Debt": "book_val",
                                              "Tangible Total Equity": "tang_book", "Debt - Total": "debt",
                                              'Total Long Term Capital': "lt_cap",
                                              'Intangible Assets - Total - Net': 'intang',
                                              "Net Cash Flow from Operating Activities": "ocf",
                                              'Capital Expenditures - Total': "capex",
                                              'Cash Flow from Operations per Share': 'cfo_ps',
                                              "Free Cash Flow to Equity": 'fcfe'})
        inc_bal = pd.concat([inc_bal, inc_bal_1])
    inc_bal['noa'] = inc_bal["tot_assets"] - inc_bal["oth_assets"]
    inc_bal['ebit_bv'] = inc_bal['ebit'] / inc_bal["book_val"]
    inc_bal['ebit_nonop'] = inc_bal['ebit'] / inc_bal["oth_assets"]
    inc_bal['ebit_op'] = inc_bal['ebit'] / inc_bal['noa']
    inc_bal['ebit_tot'] = inc_bal['ebit'] / inc_bal["tot_assets"]
    inc_bal['ebit_mgn'] = inc_bal['ebit'] / inc_bal['rev']
    inc_bal['net_debt'] = inc_bal['debt'] - inc_bal['cash']
    inc_bal['nd_ebitda'] = inc_bal['net_debt'] / inc_bal['ebitda']
    inc_bal['ni_op'] = inc_bal['net_inc'] / inc_bal['noa']
    inc_bal['ni_tot'] = inc_bal['net_inc'] / inc_bal["tot_assets"]
    inc_bal['ni_mgn'] = inc_bal['net_inc'] / inc_bal['rev']
    inc_bal['ocf_bv'] = inc_bal['ocf'] / inc_bal["book_val"]
    inc_bal['ocf_op'] = inc_bal['ocf'] / inc_bal['noa']
    inc_bal['ocf_tot'] = inc_bal['ocf'] / inc_bal["tot_assets"]
    inc_bal['ocf_mgn'] = inc_bal['ocf'] / inc_bal['rev']
    inc_bal['ocf_ce'] = inc_bal['ocf'] / inc_bal['lt_cap']
    inc_bal['ocf_bv'] = inc_bal['ocf'] / inc_bal["book_val"]
    inc_bal['ocf_op'] = inc_bal['ocf'] / inc_bal['noa']
    inc_bal['ocf_tot'] = inc_bal['ocf'] / inc_bal["tot_assets"]
    inc_bal['ocf_mgn'] = inc_bal['ocf'] / inc_bal['rev']
    inc_bal['ocf_ce'] = inc_bal['ocf'] / inc_bal['lt_cap']
    inc_bal['fcf_bv'] = inc_bal['fcfe'] / inc_bal["book_val"]
    inc_bal['fcf_op'] = inc_bal['fcfe'] / inc_bal['noa']
    inc_bal['fcf_tot'] = inc_bal['fcfe'] / inc_bal["tot_assets"]
    inc_bal['fcf_mgn'] = inc_bal['fcfe'] / inc_bal['rev']
    inc_bal['fcf_ce'] = inc_bal['fcfe'] / inc_bal['lt_cap']
    inc_bal['fcf_bv'] = inc_bal['fcfe'] / inc_bal["book_val"]
    inc_bal['fcf_op'] = inc_bal['fcfe'] / inc_bal['noa']
    inc_bal['fcf_tot'] = inc_bal['fcfe'] / inc_bal["tot_assets"]
    inc_bal['fcf_mgn'] = inc_bal['fcfe'] / inc_bal['rev']
    inc_bal['fcf_ce'] = inc_bal['fcfe'] / inc_bal['lt_cap']
    inc_bal.to_csv('assets/models/jeff_multi_factor/accounting_feats.csv', index=False)

    return inc_bal


def mkt_cap_feat():
    mkt_cap = pd.read_csv("assets/models/jeff_multi_factor/mkt_cap.csv", index_col=0)
    mkt_cap.columns = [re.findall("(?<=\<)(.*?)(?=\>)", i)[0] for i in mkt_cap.columns]
    mkt_cap12m = mkt_cap.rolling(252).mean().dropna(axis=0, how='all')
    mkt_cap12m = mkt_cap12m.rename(columns={i: '{}_12m_avg_mktcap'.format(i) for i in mkt_cap12m.columns})
    mkt_cap3m = mkt_cap.rolling(63).mean().dropna(axis=0, how='all')
    mkt_cap3m = mkt_cap3m.rename(columns={i: '{}_3m_avg_mktcap'.format(i) for i in mkt_cap3m.columns})
    mkt_cap6m = mkt_cap.rolling(126).mean().dropna(axis=0, how='all')
    mkt_cap6m = mkt_cap6m.rename(columns={i: '{}_6m_avg_mktcap'.format(i) for i in mkt_cap6m.columns})
    mkt_cap_features = mkt_cap12m.join(mkt_cap6m, how='inner')
    mkt_cap_features = mkt_cap_features.join(mkt_cap3m, how='inner')
    mkt_cap_features.to_csv('assets/models/jeff_multi_factor/mkt_cap_feats.csv')
    return mkt_cap_features


def merge_features():
    mkt_cap = pd.read_csv("assets/models/jeff_multi_factor/mkt_cap_feats.csv", index_col=0)
    vol = pd.read_csv("assets/models/jeff_multi_factor/vol_df.csv", index_col=0)
    trading = mkt_cap.join(vol, how='inner')
    trading.index.name = 'date'
    trading.index = pd.to_datetime(trading.index)
    accounting = pd.read_csv("assets/models/jeff_multi_factor/accounting_feats.csv")
    accounting['date'] = pd.to_datetime(accounting['date']).dt.date
    accounting = accounting.set_index('date')
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    df = pd.DataFrame()
    for r in rics:
        try:
            act = accounting[accounting['Instrument'] == r]
            vol_12 = trading['{}_12m_volume'.format(r)].to_frame(name='12m_volume')
            vol_6 = trading['{}_6m_volume'.format(r)].to_frame(name='6m_volume')
            vol_3 = trading['{}_3m_volume'.format(r)].to_frame(name='3m_volume')
            mkt_12 = trading['{}_12m_avg_mktcap'.format(r)].to_frame(name='12m_avg_mkt_cap')
            mkt_6 = trading['{}_6m_avg_mktcap'.format(r)].to_frame(name='6m_avg_mkt_cap')
            mkt_3 = trading['{}_3m_avg_mktcap'.format(r)].to_frame(name='3m_avg_mkt_cap')
            trad = vol_12.join(vol_6, how='inner')
            trad = trad.join(vol_3, how='inner')
            trad = trad.join(mkt_12, how='inner')
            trad = trad.join(mkt_3, how='inner')
            trad = trad.join(mkt_6, how='inner')
            trad = trad.sort_index(ascending=False)
            d = act.join(trad, how='inner')
            df = pd.concat([df, d])
        except:
            pass
    val_mo = pd.read_csv("assets/fundamentals/production.csv")
    val_mo = val_mo.rename(columns={'Date': 'date'})
    val_mo['date'] = pd.to_datetime(val_mo['date']).dt.date
    tics = [i.split('.')[0] for i in list(df['Instrument'])]
    tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics = ['BF-B' if i == 'BFb' else i for i in tics]
    df = df.reset_index()
    df['Instrument'] = tics
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.merge(val_mo, on=['date', 'Instrument'], how='outer')
    qual = pd.read_csv('qual_dat.csv')
    tics = [i.split('.')[0] for i in list(qual['Instrument'])]
    tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics = ['BF-B' if i == 'BFb' else i for i in tics]
    qual["Instrument"] = tics
    qual = qual.rename(columns={"Original Announcement Date Time": 'date',
                                "Return on Average Common Equity - %, TTM": "roe",
                                "Return on Average Total Assets - %, TTM": "roa",
                                "Return on Average Total Long Term Capital - %, TTM": "roce",
                                "Return on Invested Capital - %, TTM": "roic",
                                "Total Debt Percentage of Total Equity": 'd_e'})
    qual['date'] = pd.to_datetime(qual['date']).dt.date
    df = df.merge(qual, on=['date', 'Instrument'], how='outer')
    return df.sort_values(['Instrument', 'date'])





def add_labs():
    t = merge_features()
    stocks = t['Instrument'].unique()
    df = pd.DataFrame()
    timer = Event()
    for s in stocks:
        try:
            px = si.get_data(s)
            ret_1yr = px['adjclose'].pct_change(252).to_frame(name='1yr_ret').shift(-252)
            ret_3m = px['adjclose'].pct_change(63).to_frame(name='3m_ret').shift(-63)
            ret_6m = px['adjclose'].pct_change(126).to_frame(name='6m_ret').shift(-126)
            vol_1yr = px['adjclose'].pct_change().rolling(252).std().to_frame(name='1yr_vol')
            vol_3m = px['adjclose'].pct_change().rolling(63).std().to_frame(name='3mth_vol')
            vol_6m = px['adjclose'].pct_change().rolling(126).std().to_frame(name="6mth_vol")
            px_based = ret_1yr.join(ret_3m, how='inner')
            px_based = px_based.join(ret_6m, how='inner')
            px_based = px_based.join(vol_1yr, how='inner')
            px_based = px_based.join(vol_6m)
            px_based = px_based.join(vol_3m)
            feats = t[t['Instrument'] == s]
            feats = feats.set_index('date').join(px_based, how='inner')
            df = pd.concat([df, feats])
            print('{}: no {} of {} complete'.format(s, len(df['Instrument'].unique()), len(stocks)))
        except:
            timer.wait(5)
            px = si.get_data(s)
            ret_1yr = px['adjclose'].pct_change(252).to_frame(name='1yr_ret').shift(-252)
            ret_3m = px['adjclose'].pct_change(63).to_frame(name='3m_ret').shift(-63)
            ret_6m = px['adjclose'].pct_change(126).to_frame(name='6m_ret').shift(-126)
            vol_1yr = px['adjclose'].pct_change().rolling(252).std().to_frame(name='1yr_vol')
            vol_3m = px['adjclose'].pct_change().rolling(63).std().to_frame(name='3mth_vol')
            vol_6m = px['adjclose'].pct_change().rolling(126).std().to_frame(name="6mth_vol")
            px_based = ret_1yr.join(ret_3m, how='inner')
            px_based = px_based.join(ret_6m, how='inner')
            px_based = px_based.join(vol_1yr, how='inner')
            px_based = px_based.join(vol_6m)
            px_based = px_based.join(vol_3m)
            feats = t[t['Instrument'] == s]
            feats = feats.set_index('date').join(px_based, how='inner')
            df = pd.concat([df, feats])
            print('{}: no {} of {} complete'.format(s, len(df['Instrument'].unique()), len(stocks)))
    df.to_csv('assets/models/jeff_multi_factor/raw_feat_lab.csv')
    return df

def partition():
    """takes raw data and preps for ml process"""
    raw = pd.read_csv('assets/models/jeff_multi_factor/raw_feat_lab.csv')
    raw = raw.rename(columns={'Unnamed: 0':'date'})
    raw['date'] = pd.to_datetime(raw['date'])
    quarts = list(raw['date'].dt.quarter)
    yrs = raw['date'].dt.year
    tfs = ['{}_{}'.format(quarts[i], yrs[i]) for i in range(0, len(quarts))]
    raw['quarter'] = tfs
    stripped = raw.dropna()
    for q in stripped['quarter'].unique():
        part = stripped[stripped['quarter']==q]
        part.to_csv('assets/models/jeff_multi_factor/{}_dat.csv'.format(q))
    return stripped.sort_values(by='quarter')
y = partition()

