import pickle
import re
from threading import Event

import DatastreamDSWS as DSWS
import eikon as ek
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si

import config

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
                                              'EPS - Basic - excl Extraordinary Items, Common - Total': 'eps_excl_basic',
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
                                              'Book Value per Share': 'bvps',
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
    qual = pd.read_csv('assets/models/jeff_multi_factor/qual_dat.csv', index_col=0)
    qual = qual.rename(columns={"Original Announcement Date Time": 'date',
                                "Return on Average Common Equity - %, TTM": "roe",
                                "Return on Average Total Assets - %, TTM": "roa",
                                "Return on Average Total Long Term Capital - %, TTM": "roce",
                                "Return on Invested Capital - %, TTM": "roic",
                                "Total Debt Percentage of Total Equity": 'd_e'})
    qual['date'] = pd.to_datetime(qual['date']).dt.date
    inc_bal['date'] = pd.to_datetime(inc_bal['date']).dt.date
    inc_bal = inc_bal.merge(qual, on=['date', 'Instrument'], how='outer')
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


def merge_vol_mkt():
    mkt_cap = pd.read_csv("assets/models/jeff_multi_factor/mkt_cap_feats.csv", index_col=0)
    vol = pd.read_csv("assets/models/jeff_multi_factor/vol_df.csv", index_col=0)
    trading = mkt_cap.join(vol, how='inner')
    trading.index.name = 'date'
    trading.index = pd.to_datetime(trading.index).date
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    df = pd.DataFrame()
    for r in rics:
        try:
            vol_12 = trading['{}_12m_volume'.format(r)].to_frame(name='12m_volume')
            vol_6 = trading['{}_6m_volume'.format(r)].to_frame(name='6m_volume')
            vol_3 = trading['{}_3m_volume'.format(r)].to_frame(name='3m_volume')
            mkt_12 = trading['{}_12m_avg_mktcap'.format(r)].to_frame(name='12m_avg_mkt_cap')
            mkt_6 = trading['{}_6m_avg_mktcap'.format(r)].to_frame(name='6m_avg_mkt_cap')
            mkt_3 = trading['{}_3m_avg_mktcap'.format(r)].to_frame(name='3m_avg_mkt_cap')
            trad = vol_12.join(vol_6, how='outer')
            trad = trad.join(vol_3, how='outer')
            trad = trad.join(mkt_12, how='outer')
            trad = trad.join(mkt_3, how='outer')
            trad = trad.join(mkt_6, how='outer')
            trad = trad.sort_index(ascending=False)
            trad['Instrument'] = r
            df = pd.concat([df, trad])
        except:
            pass
    df.to_csv('assets/models/jeff_multi_factor/mkt_vol_dat.csv')
    return df


def creat_labs_vol():
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    tics = [i.split('.')[0] for i in rics]
    tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics = ['BF-B' if i == 'BFb' else i for i in tics]
    df = pd.DataFrame()
    px_dat = pd.DataFrame()
    timer = Event()
    for s in tics:
        try:
            px = si.get_data(s)
            px = px.rename(columns={"ticker": "Instrument"})
            prices = px.filter(['Instrument', 'close'])
            px_dat = pd.concat([px_dat, prices])
            ret_1yr = px['adjclose'].pct_change(252).to_frame(name='1yr_ret').shift(-252)
            ret_3m = px['adjclose'].pct_change(63).to_frame(name='3m_ret').shift(-63)
            ret_6m = px['adjclose'].pct_change(126).to_frame(name='6m_ret').shift(-126)
            vol_1yr = px['adjclose'].pct_change().rolling(252).std().to_frame(name='1yr_vol') * np.sqrt(252)
            vol_3m = px['adjclose'].pct_change().rolling(63).std().to_frame(name='3mth_vol') * np.sqrt(252)
            vol_6m = px['adjclose'].pct_change().rolling(126).std().to_frame(name="6mth_vol") * np.sqrt(252)
            mom_1yr = px['adjclose'].pct_change(252).to_frame(name='1yr_mom')
            mom_3m = px['adjclose'].pct_change(63).to_frame(name='3m_mom')
            mom_6m = px['adjclose'].pct_change(126).to_frame(name='6m_mom')
            px_based = ret_1yr.join(ret_3m, how='outer')
            px_based = px_based.join(ret_6m, how='outer')
            px_based = px_based.join(vol_1yr, how='outer')
            px_based = px_based.join(vol_6m)
            px_based = px_based.join(vol_3m)
            px_based = px_based.join(mom_1yr)
            px_based = px_based.join(mom_6m)
            px_based = px_based.join(mom_3m)
            px_based['Instrument'] = s
            df = pd.concat([df, px_based])
            print('{}: no {} of {} complete'.format(s, len(df['Instrument'].dropna().unique()), len(tics)))
        except:
            timer.wait(5)
            px = si.get_data(s)
            px = px.rename(columns={"ticker": "Instrument"})
            prices = px.filter(['Instrument', 'close'])
            px_dat = pd.concat([px_dat, prices])
            ret_1yr = px['adjclose'].pct_change(252).to_frame(name='1yr_ret').shift(-252)
            ret_3m = px['adjclose'].pct_change(63).to_frame(name='3m_ret').shift(-63)
            ret_6m = px['adjclose'].pct_change(126).to_frame(name='6m_ret').shift(-126)
            vol_1yr = px['adjclose'].pct_change().rolling(252).std().to_frame(name='1yr_vol') * np.sqrt(252)
            vol_3m = px['adjclose'].pct_change().rolling(63).std().to_frame(name='3mth_vol') * np.sqrt(252)
            vol_6m = px['adjclose'].pct_change().rolling(126).std().to_frame(name="6mth_vol") * np.sqrt(252)
            mom_1yr = px['adjclose'].pct_change(252).to_frame(name='1yr_mom')
            mom_3m = px['adjclose'].pct_change(63).to_frame(name='3m_mom')
            mom_6m = px['adjclose'].pct_change(126).to_frame(name='6m_mom')
            px_based = ret_1yr.join(ret_3m, how='outer')
            px_based = px_based.join(ret_6m, how='outer')
            px_based = px_based.join(vol_1yr, how='outer')
            px_based = px_based.join(vol_6m)
            px_based = px_based.join(vol_3m)
            px_based = px_based.join(mom_1yr)
            px_based = px_based.join(mom_6m)
            px_based = px_based.join(mom_3m)
            px_based['Instrument'] = s
            df = pd.concat([df, px_based])
            print('{}: no {} of {} complete'.format(s, len(df['Instrument'].dropna().unique()), len(tics)))
    px_dat.to_csv('assets/models/jeff_multi_factor/close_prices.csv')
    df.to_csv("assets/models/jeff_multi_factor/vol_labs.csv")
    return df


def valuation():
    df = pd.read_csv('assets/models/jeff_multi_factor/accounting_feats.csv')
    df = df.rename(columns={'Unnamed: 0': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.date
    rics = list(df['Instrument'])
    tics = [i.split('.')[0] for i in rics]
    tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics = ['BF-B' if i == 'BFb' else i for i in tics]
    df['Instrument'] = tics
    df = df.dropna(subset=['date'])
    acts = df.filter(['Instrument', 'date', 'eps_excl_basic', 'bvps', 'cfo_ps', 'ebit',
                      'ebitda', 'rev', 'debt', 'cash'])
    px = pd.read_csv('assets/models/jeff_multi_factor/close_prices.csv')
    px = px.rename(columns={'Unnamed: 0': 'date', 'ticker': 'Instrument'})
    px['date'] = pd.to_datetime(px['date']).dt.date
    valuation_df = px.merge(acts, on=['date', 'Instrument'], how='outer').fillna(method='ffill')
    valuation_df = valuation_df.drop_duplicates()
    valuation_df['p_e'] = valuation_df['close'] / valuation_df["eps_excl_basic"]
    valuation_df['p_b'] = valuation_df['close'] / valuation_df["bvps"]
    valuation_df['p_cf'] = valuation_df['close'] / valuation_df["cfo_ps"]
    mkt_cap = pd.read_csv("assets/models/jeff_multi_factor/mkt_cap.csv", index_col=0)
    mkt_cap.columns = [re.findall("(?<=\<)(.*?)(?=\>)", i)[0] for i in mkt_cap.columns]
    mkt_cap.columns = [i.split('.')[0] for i in mkt_cap.columns]
    mkt_cap = mkt_cap.reset_index()
    mkt_cap = mkt_cap.rename(columns={'Dates': 'date'})
    mkt_cap['date'] = pd.to_datetime(mkt_cap['date']).dt.date
    mkt_values = pd.melt(mkt_cap, id_vars='date', var_name='Instrument', value_name='mkt_cap')
    valuation_df = valuation_df.merge(mkt_values, on=['date', 'Instrument'], how='outer').dropna()
    valuation_df['ev'] = valuation_df['mkt_cap'] + valuation_df['debt'] - valuation_df['cash']
    valuation_df['ev_ebit'] = valuation_df['ev'] / valuation_df['ebit']
    valuation_df['ev_ebitda'] = valuation_df['ev'] / valuation_df['ebitda']
    valuation_df['ev_sales'] = valuation_df['ev'] / valuation_df['rev']
    valuation_df = valuation_df.filter(['date', 'Instrument', 'p_e', 'p_b', 'p_cf', 'ev',
                                        'ev_ebit', 'ev_ebitda', 'ev_sales'])
    valuation_df.to_csv('assets/models/jeff_multi_factor/valuation.csv')
    return valuation_df


def technicals():
    px = pd.read_csv('assets/models/jeff_multi_factor/close_prices.csv')
    px = px.rename(columns={'Unnamed: 0': 'date', 'ticker': 'Instrument'})
    px['date'] = pd.to_datetime(px['date']).dt.date
    df = pd.DataFrame()
    for s in px['Instrument'].unique():
        stock = px[px['Instrument'] == s]
        stock['200_ma'] = stock['close'].ewm(span=200).mean()
        stock['50_ma'] = stock['close'].ewm(span=50).mean()
        df = pd.concat([df, stock])
    df.to_csv('assets/models/jeff_multi_factor/moving_av.csv')
    return df


def merge_data():
    mkt_vol = pd.read_csv('assets/models/jeff_multi_factor/mkt_vol_dat.csv')
    mkt_vol = mkt_vol.rename(columns={'Unnamed: 0': 'date'})
    rics = list(mkt_vol['Instrument'])
    tics = [i.split('.')[0] for i in rics]
    tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics = ['BF-B' if i == 'BFb' else i for i in tics]
    mkt_vol['Instrument'] = tics
    mkt_vol['date'] = pd.to_datetime(mkt_vol['date']).dt.date
    mom_labs = pd.read_csv("assets/models/jeff_multi_factor/vol_labs.csv")
    mom_labs = mom_labs.rename(columns={'Unnamed: 0': 'date'})
    mom_labs['date'] = pd.to_datetime(mom_labs['date']).dt.date
    labs = mom_labs.filter(['Instrument', 'date', '1yr_ret', '3m_ret', '6m_ret'])
    mom = mom_labs.drop(['1yr_ret', '3m_ret', '6m_ret'], axis=1)
    val = pd.read_csv('assets/models/jeff_multi_factor/valuation.csv')
    val['date'] = pd.to_datetime(val['date']).dt.date
    tech = pd.read_csv('assets/models/jeff_multi_factor/moving_av.csv', index_col=0)
    tech['date'] = pd.to_datetime(tech['date']).dt.date
    act = pd.read_csv('assets/models/jeff_multi_factor/accounting_feats.csv')
    rics = list(act['Instrument'])
    tics = [i.split('.')[0] for i in rics]
    tics = ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics = ['BF-B' if i == 'BFb' else i for i in tics]
    act['Instrument'] = tics
    act['date'] = pd.to_datetime(act['date']).dt.date
    data = mkt_vol.merge(mom, on=['date', 'Instrument'], how='inner')
    data = data.merge(val, on=['date', 'Instrument'], how='inner')
    data = data.merge(tech, on=['date', 'Instrument'], how='inner')
    data['date'] = pd.to_datetime(data['date']).dt.date
    data = data.merge(act, on=['Instrument', 'date'], how='outer')
    data = data.sort_values(by=['Instrument', 'date'])
    data = data.drop_duplicates()
    data = data.fillna(method='ffill').dropna()
    labs = labs.sort_values(by=['Instrument', 'date'])
    data = data.merge(labs, on=['Instrument', 'date'], how='outer')
    data.to_csv('assets/models/jeff_multi_factor/aggregate_features.csv')

    return data


inc_bal = combine_inc_bal()
mkt_feat = mkt_cap_feat()
trad = merge_vol_mkt()
labs_vol = creat_labs_vol()
val = valuation()
tech = technicals()
data = merge_data()
data = pd.read_csv('assets/models/jeff_multi_factor/aggregate_features.csv', index_col=0)

