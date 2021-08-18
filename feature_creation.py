import os
import time
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import glob
from tqdm import tqdm
import yfinance as yf
from yahoo_fin.stock_info import *
import pickle
from threading import Event
from sqlalchemy import create_engine

# Currently not used, as the ta library is used to provide all technical features
def daily_features():
    path = r'assets/historical-symbols'  # use your path
    all_files = glob.glob(path + "/*.csv")

    # Creating list to append all ticker dfs to
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    # Concat all ticker dfs
    stock_df = pd.concat(li, axis=0, ignore_index=True, sort=True)

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Creating Moving Average Technical Indicator
    # Using this article
    # https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for quantitative-trading-8d98751b5fb
    stock_df['SMA_5'] = stock_df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=5).mean())
    stock_df['SMA_15'] = stock_df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=15).mean())
    stock_df['SMA_ratio'] = stock_df['SMA_15'] / stock_df['SMA_5']

    # Bollinger bands
    stock_df['SD'] = stock_df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=15).std())
    stock_df['upperband'] = stock_df['SMA_15'] + 2 * stock_df['SD']
    stock_df['lowerband'] = stock_df['SMA_15'] - 2 * stock_df['SD']

    # Creating datetime date and making index
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.index = stock_df['Date']
    stock_df.drop('Date', axis='columns', inplace=True)

    return stock_df


###sql engine 
engine = create_engine('postgresql+psycopg2://postgres:poRter!5067@databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com:5432/697_temp')




def get_spy_stocks_pkl():
    """This function imports the Symbols (RICS) for all stocks in the SPDR SP 500 ETF do not run w/o a
    subscription to refinitive. """
    date = datetime.now().date().strftime('%Y-%m-%d')
    hold = ek.get_data('SPY', fields=[ek.TR_Field('TR.ETPConstituentRIC', params={'SDate': date})])[0]
    hold = hold[hold['Constituent RIC'] != 'GOOG.OQ']
    rics = [x for x in hold['Constituent RIC']]
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "wb") as f:
        pickle.dump(rics, f)
    return





def batch(iterable, n=1):
    """
    Helper function that assist in managing batch sizes
    :param iterable: an iterable python object
    :param n: int - batch size 
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



def create_volume_features():
    """
    Retrives daily trading volume data do not use without a subscription to Eikon
    """
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    vol_df = pd.DataFrame()
    timer = Event()
    for x in batch(rics, 2):
        try:
            q_1 = ek.get_timeseries(x, fields="VOLUME", start_date=get_date_from_today(1095), interval='daily')
            vol_df = vol_df.join(q_1, how='outer')
        except:
            timer.wait(10)
            q_1 = ek.get_timeseries(x, fields="VOLUME", start_date=get_date_from_today(1095), interval='daily')
            vol_df = vol_df.join(q_1, how='outer')
        if len(vol_df.columns)%50==0:
            print(len(vol_df.columns))
    vol_df = vol_df.dropna(axis=1)
    vol_12m = vol_df.rolling(252).mean().dropna(axis=0, how='all')
    vol_12 = vol_12m.rename(columns={i: '{}_12m_volume'.format(i) for i in vol_12m.columns})
    vol_3m = vol_df.rolling(63).mean().dropna(axis=0, how='all')
    vol_3 = vol_3m.rename(columns={i: '{}_3m_volume'.format(i) for i in vol_3m.columns})
    vol_6m = vol_df.rolling(126).mean().dropna(axis=0, how='all')
    vol_6 = vol_6m.rename(columns={i: '{}_6m_volume'.format(i) for i in vol_6m.columns})
    dfs = [vol_12, vol_6, vol_3]
    table_names = ["vol_12", "vol_6", "vol_3"]
    for i in range(0, len(dfs)):
        dfs[i].reset_index().to_sql('{}'.format(table_names[i]), con = engine, if_exists = 'replace', index = False)
        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        dfs[i].to_csv("assets/models/jeff_multi_factor/{}_df.csv".format(table_names[i]), sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        cur.copy_from(output, table_names[i], null="") # null values become ''
        conn.commit()
        os.remove("assets/models/jeff_multi_factor/{}_df.csv".format(table_names[i]))
    
    print('Done!')
    return




def get_income_stat_dat():
    """"
    Retrieves income statement data for each S&P 500 company do not use without subscription to Eikon
    """
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    is_dat = pd.DataFrame()
    timer = Event()
    for i in batch(rics, 40):
        try:
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.EPSBasicInclExordItemsComTot",
                                       "TR.F.EPSBasicExclExordItemsComTot", "TR.F.EPSDilExclExordItemsComTot",
                                       "TR.F.EPSBasicInclExOrdComTotPoPDiff", "TR.F.EBIT", "TR.F.EBITDA",
                                       "TR.F.TotRevenue", "TR.F.IncAvailToComShr"],
                            parameters={"Period": "LTM", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            is_dat = pd.concat([is_dat, t])
        except:
            timer.wait(10)
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.EPSBasicInclExordItemsComTot",
                                       "TR.F.EPSBasicExclExordItemsComTot", "TR.F.EPSDilExclExordItemsComTot",
                                       "TR.F.EPSBasicInclExOrdComTotPoPDiff", "TR.F.EBIT", "TR.F.EBITDA",
                                       "TR.F.TotRevenue", "TR.F.IncAvailToComShr"],
                            parameters={"Period": "LTM", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            is_dat = pd.concat([is_dat, t])
    is_dat.to_sql("inc_stat", con = engine, if_exists = 'replace', index = False)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    is_dat.to_csv('assets/models/jeff_multi_factor/income_stat_dat.csv', sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, "inc_stat")
    conn.commit()
    os.remove("assets/models/jeff_multi_factor/income_stat_dat.csv")    
    print("done")
    return




def get_bal_sheet_dat():
    """
    Retrieves balance sheet data for each S&P 500 company do not use without a subscription to Eikon
    """
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    bs_dat = pd.DataFrame()
    timer = Event()
    for i in batch(rics, 40):
        try:
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.TotAssets", "TR.F.OthAssetsTot",
                                       "TR.F.CashSTInvstTot", "TR.F.TotShHoldEq", "TR.F.TangTotEq", "TR.F.DebtTot",
                                       "TR.F.TotLTCap", "TR.F.IntangTotNet", "TR.F.BookValuePerShr"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            bs_dat = pd.concat([bs_dat, t])
        except:
            timer.wait(10)
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.TotAssets", "TR.F.OthAssetsTot",
                                       "TR.F.CashSTInvstTot", "TR.F.TotShHoldEq", "TR.F.TangTotEq", "TR.F.DebtTot",
                                       "TR.F.TotLTCap", "TR.F.IntangTotNet", "TR.F.BookValuePerShr"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            bs_dat = pd.concat([bs_dat, t])
    bs_dat.to_sql("bal_sht", con = engine, if_exists = 'replace', index = False)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    bs_dat.to_csv('assets/models/jeff_multi_factor/bal_sht_dat.csv', sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, "bal_sht")
    conn.commit()
    os.remove('assets/models/jeff_multi_factor/bal_sht_dat.csv')    
    print("done")

    return




def get_cf_dat():
    """"
    Retrieves cashflow data for all S&P 500 Companies, do not use without subscriotion to Eikon
    """
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    cf_dat = pd.DataFrame()
    timer = Event()
    for i in batch(rics, 40):
        try:
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.NetCashFlowOp", "TR.F.CAPEXTot",
                                       "TR.F.NetCFOpPerShr", "TR.F.FreeCashFlowToEq"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            cf_dat = pd.concat([cf_dat, t])
        except:
            timer.wait(10)
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.NetCashFlowOp", "TR.F.CAPEXTot",
                                       "TR.F.NetCFOpPerShr", "TR.F.FreeCashFlowToEq"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            cf_dat = pd.concat([cf_dat, t])
    cf_dat.to_sql("cf_stat", con = engine, if_exists = 'replace', index = False)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    cf_dat.to_csv('assets/models/jeff_multi_factor/cf_dat.csv', sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, "cf_stat")
    conn.commit()
    os.remove("assets/models/jeff_multi_factor/cf_dat.csv")    
    print("done")
    return




def get_qual_dat():
    """
    Retrieves quality and profitability data for all S&P 500 companies - do not use without subscription to Eikon
    """
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    ql_dat = pd.DataFrame()
    timer = Event()
    for i in batch(rics, 40):
        try:
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.ReturnAvgComEqPctTTM",
                                       "TR.F.ReturnAvgTotAssetsPctTTM", "TR.F.ReturnAvgTotLTCapPctTTM",
                                       "TR.F.ReturnInvstCapPctTTM", "TR.F.TotDebtPctofTotEq"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            ql_dat = pd.concat([ql_dat, t])
        except:
            timer.wait(10)
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.ReturnAvgComEqPctTTM",
                                       "TR.F.ReturnAvgTotAssetsPctTTM", "TR.F.ReturnAvgTotLTCapPctTTM",
                                       "TR.F.ReturnInvstCapPctTTM", "TR.F.TotDebtPctofTotEq"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            ql_dat = pd.concat([ql_dat, t])
    ql_dat.to_sql("qual_dat", con = engine, if_exists = 'replace', index = False)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    ql_dat.to_csv('assets/models/jeff_multi_factor/qual_dat.csv', sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, "qual_dat")
    conn.commit()
    os.remove('assets/models/jeff_multi_factor/qual_dat.csv')    
    print("done")
    return




def get_mv():
    """
    retrieves market cap data for all S&P 500 companies do not use without a subscription to datastream
    """
    timer = Event()
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    df = pd.DataFrame()
    for i in batch(rics, 2):
        get = '<{}>,<{}>'.format(i[0], i[1])
        try:
            data = ds.get_data(tickers=get, fields=['MV'], start="2017-12-31", freq='D')
            df = df.join(data, how='outer')
            if len(df.columns)%50 ==0:
                print(len(df.columns))
        except:
            timer.wait(10)
            data = ds.get_data(tickers=get, fields=['MV'], start="2017-12-31", freq='D')
            df = df.join(data, how='outer')
            if len(df.columns)%50 ==0:
                print(len(df.columns))
    df = df.reset_index()
    df.to_sql("mkt_cap", con = engine, if_exists = 'replace', index = False)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv('assets/models/jeff_multi_factor/mkt_cap.csv', sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, "mkt_cap")
    conn.commit()
    os.remove('assets/models/jeff_multi_factor/mkt_cap.csv')    
    print("done")
    return 