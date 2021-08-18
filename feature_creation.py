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


def get_spy_stocks():
    """This function imports the Symbols (RICS) for all stocks in the SPDR SP 500 ETF do not run w/o a
    subscription to refinitive. """
    date = datetime.now().date().strftime('%Y-%m-%d')
    hold = ek.get_data('SPY', fields=[ek.TR_Field('TR.ETPConstituentRIC', params={'SDate': date})])[0]
    hold = hold[hold['Constituent RIC'] != 'GOOG.OQ']
    rics = [x for x in hold['Constituent RIC']]
    return rics


def get_rev_dat(stocks):
    """Gets Revenue data for a list of stocks and outputs a df do not run w/o a subscription to refinitive."""
    fields = [
        ek.TR_Field('TR.F.OriginalAnnouncementDate', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'}),
        ek.TR_Field('TR.F.TotRevBizActiv', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'})]
    t = np.linspace(0, len(stocks), num=5, dtype='int')

    rev_df = pd.DataFrame()
    count = 0
    for i in tqdm(range(0, len(t))):
        try:
            df = ek.get_data(stocks[t[i]:t[i + 1]], fields=fields)[0]
        except:
            df = ek.get_data(stocks[t[i]:], fields=fields)[0]

        rev_df = pd.concat([rev_df, df])
        count += 1

    rev_df = rev_df.drop_duplicates()
    rev_df.to_csv('assets/fundamentals/Revenue_df.csv', index=False)
    return rev_df


def get_inc_dat(stocks):
    """Gets Net income after taxes data for a list of stocks and outputs a df do not run w/o a subscription to refinitive."""
    fields = [ek.TR_Field('TR.F.IncAvailToComShr', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'}),
              ek.TR_Field('TR.F.OriginalAnnouncementDate',
                          params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'})]
    t = np.linspace(0, len(stocks), num=5, dtype='int')

    inc_df = pd.DataFrame()
    count = 0
    for i in tqdm(range(0, len(t))):
        try:
            df = ek.get_data(stocks[t[i]:t[i + 1]], fields=fields)[0]
        except:
            df = ek.get_data(stocks[t[i]:], fields=fields)[0]

        inc_df = pd.concat([inc_df, df])
        count += 1

    inc_df = inc_df.drop_duplicates()
    inc_df.to_csv('assets/fundamentals/Net_income_df.csv', index=False)
    return inc_df


def get_ebit_dat(stocks):
    """Gets Normalized EBIT data for a list of stocks and outputs a df do not run w/o a subscription to refinitive."""
    fields = [ek.TR_Field('TR.F.EBITNorm', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'}),
              ek.TR_Field('TR.F.OriginalAnnouncementDate',
                          params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'})]
    t = np.linspace(0, len(stocks), num=5, dtype='int')

    ebit_df = pd.DataFrame()
    count = 0
    for i in tqdm(range(0, len(t))):
        try:
            df = ek.get_data(stocks[t[i]:t[i + 1]], fields=fields)[0]
        except:
            df = ek.get_data(stocks[t[i]:], fields=fields)[0]

        ebit_df = pd.concat([ebit_df, df])
        count += 1

    ebit_df = ebit_df.drop_duplicates()
    ebit_df.to_csv('assets/fundamentals/EBIT_df.csv', index=False)
    return ebit_df


def get_book_dat(stocks):
    """Gets Equity to shareholders data for a list of stocks and outputs a df do not run w/o a
    subscription to refinitive."""
    fields = [ek.TR_Field('TR.F.ComEqTot', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'}),
              ek.TR_Field('TR.F.OriginalAnnouncementDate',
                          params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'})]
    t = np.linspace(0, len(stocks), num=5, dtype='int')

    book_df = pd.DataFrame()
    count = 0
    for i in tqdm(range(0, len(t))):
        try:
            df = ek.get_data(stocks[t[i]:t[i + 1]], fields=fields)[0]
        except:
            df = ek.get_data(stocks[t[i]:], fields=fields)[0]

        book_df = pd.concat([book_df, df])
        count += 1

    book_df = book_df.drop_duplicates()
    book_df.to_csv('assets/fundamentals/Book_df.csv', index=False)
    return book_df


def get_ev_dat(stocks):
    """Gets Enterprise Value data for a list of stocks and outputs a df do not run w/o a subscription to refinitive."""
    fields = [ek.TR_Field('TR.F.EV', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'}),
              ek.TR_Field('TR.F.OriginalAnnouncementDate',
                          params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'})]
    t = np.linspace(0, len(stocks), num=5, dtype='int')

    ev_df = pd.DataFrame()
    count = 0
    for i in tqdm(range(0, len(t))):
        try:
            df = ek.get_data(stocks[t[i]:t[i + 1]], fields=fields)[0]
        except:
            df = ek.get_data(stocks[t[i]:], fields=fields)[0]

        ev_df = pd.concat([ev_df, df])
        count += 1

    ev_df = ev_df.drop_duplicates()
    ev_df.to_csv('assets/fundamentals/EV_df.csv', index=False)
    return ev_df


def get_mktcap_dat(stocks):
    """Gets Market Cap data for a list of stocks and outputs a df do not run w/o a subscription to refinitive."""
    fields = [ek.TR_Field('TR.F.MktCap', params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'}),
              ek.TR_Field('TR.F.OriginalAnnouncementDate',
                          params={'SDate': 0, 'EDate': -19, 'Period': 'FQ0', 'Frq': 'FQ'})]
    t = np.linspace(0, len(stocks), num=5, dtype='int')

    mcap_df = pd.DataFrame()
    count = 0
    for i in tqdm(range(0, len(t))):
        try:
            df = ek.get_data(stocks[t[i]:t[i + 1]], fields=fields)[0]
        except:
            df = ek.get_data(stocks[t[i]:], fields=fields)[0]

        mcap_df = pd.concat([mcap_df, df])
        count += 1

    mcap_df = mcap_df.drop_duplicates()
    mcap_df.to_csv('assets/fundamentals/mktcap_df.csv', index=False)
    return mcap_df


def get_adj_px():
    """aggregates max historic adjusted prices for all stocks into one large csv"""
    stocks = pd.read_csv("symbols.csv")
    stock_list = list(stocks['Symbols'])
    df = pd.DataFrame()
    count = 0
    for stock in stock_list:
        try:
            x = yf.download(stock, period='max', interval='1d')
            x = x.filter(['Adj Close'])
            x = x.rename(columns={'Adj Close': stock})
            df = df.join(x, how='outer')
            count += 1
            print("{} of {} {} complete w/{} rows".format(count, len(stock_list), stock, len(x)))
        except:
            count += 1
            print("Failed to retrieve data for {}, {} of {}".format(stock, count, len(stock_list)))
        if count % 50 == 0:
            time.sleep(10)
    df.to_csv("assets/fundamentals/price_dat.csv")
    return df


def get_sectors():
    tickers = tickers_sp500(True)
    sectors = tickers.filter(['Symbol', 'GICS Sector'])
    sectors = sectors.rename(columns={'Symbol': 'Instrument'})
    sectors.to_csv("assets/fundamentals/sectors.csv")
    return


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
    """Helper function that assist in managing batch sizes"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def create_volume_features():
    """Retrives daily trading volume data do not use without a subscription to Eikon"""
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
        print(len(vol_df.columns))
    vol_df = vol_df.dropna(axis=1)
    vol_12m = vol_df.rolling(252).mean().dropna(axis=0, how='all')
    vol_12m = vol_12m.rename(columns={i: '{}_12m_volume'.format(i) for i in vol_12m.columns})
    vol_3m = vol_df.rolling(63).mean().dropna(axis=0, how='all')
    vol_3m = vol_3m.rename(columns={i: '{}_3m_volume'.format(i) for i in vol_3m.columns})
    vol_6m = vol_df.rolling(126).mean().dropna(axis=0, how='all')
    vol_6m = vol_6m.rename(columns={i: '{}_6m_volume'.format(i) for i in vol_6m.columns})
    vol_features = vol_12m.join(vol_6m, how='inner')
    vol_features = vol_features.join(vol_3m, how='inner')
    vol_features.to_csv("assets/models/jeff_multi_factor/vol_df.csv")
    print('Done!')
    return


def get_income_stat_dat():
    """"Retrieves income statement data for each S&P 500 company do not use without subscription to Eikon"""
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
    is_dat.to_csv('assets/models/jeff_multi_factor/income_stat_dat.csv')
    print("done")
    return


def get_bal_sheet_dat():
    """Retrieves balance sheet data for each S&P 500 company do not use without a subscription to Eikon"""
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    bs_dat = pd.DataFrame()
    timer = Event()
    for i in batch(rics, 40):
        try:
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.TotAssets", "TR.F.OthAssetsTot",
                                       "TR.F.CashSTInvstTot", "TR.F.TotShHoldEq", "TR.F.TangTotEq", "TR.F.DebtTot",
                                       "TR.F.TotLTCap", "TR.F.IntangTotNet"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            bs_dat = pd.concat([bs_dat, t])
        except:
            timer.wait(10)
            t = ek.get_data(i, fields=["TR.F.OriginalAnnouncementDate", "TR.F.TotAssets", "TR.F.OthAssetsTot",
                                       "TR.F.CashSTInvstTot", "TR.F.TotShHoldEq", "TR.F.TangTotEq", "TR.F.DebtTot",
                                       "TR.F.TotLTCap", "TR.F.IntangTotNet"],
                            parameters={"Period": "FQ0", "Frq": "FQ", "SDate": 0, "Edate": -7})[0]
            bs_dat = pd.concat([bs_dat, t])
    bs_dat.to_csv('assets/models/jeff_multi_factor/bal_sht_dat.csv')
    print("done")
    return


def get_cf_dat():
    """"Retrieves cashflow data for all S&P 500 Companies, do not use without subscriotion to Eikon"""
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
    cf_dat.to_csv('assets/models/jeff_multi_factor/cf_dat.csv')
    print("done")
    return


def get_qual_dat():
    """Retrieves quality and profitability data for all S&P 500 companies - do not use without subscription to Eikon"""
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
    ql_dat.to_csv('assets/models/jeff_multi_factor/qual_dat.csv')
    print("done")
    return

def get_mv():
    """retrieves market cap data for all S&P 500 companies do not use without a subscription to datastream """
    timer = Event()
    with open("assets/models/jeff_multi_factor/spy_rics.pkl", "rb") as f:
        rics = pickle.load(f)
    df = pd.DataFrame()
    for i in batch(rics, 2):
        get = '<{}>,<{}>'.format(i[0], i[1])
        try:
            data = ds.get_data(tickers=get, fields=['MV'], start="2017-12-31", freq='D')
            df = df.join(data, how='outer')
            print(len(df.columns))
        except:
            timer.wait(10)
            data = ds.get_data(tickers=get, fields=['MV'], start="2017-12-31", freq='D')
            df = df.join(data, how='outer')
            print(len(df.columns))
    df.to_csv('assets/models/jeff_multi_factor/mkt_cap.csv')
    return df