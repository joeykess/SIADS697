import pandas as pd
import DatastreamDSWS as DSWS
import config
ds = DSWS.Datastream(username = config.username_ds(), password=config.pw_ds())

def feature_prep():
    raw = pd.read_csv("assets/macro/raw_macro.csv")
    raw.columns=raw.columns.rename('')
    raw = raw.iloc[2:]
    raw = raw.rename(columns = {"Instrument": "date", "USLIAWHMP": "avg_weekly_hrs", " USUNINSCE":"init_claims",
                                " USCNORCGD":"cons_goods_new_ords", " USNAPMNO":"ism_new_ords",
                                " USNOEXCHD": "non_def_cap"," USBPPRVTO":"buld_permits", " S&PCOMP":"sp_500",
                                " USBCILCIQ": "credit", " FRTCM10": "ust_10yr", " FRFEDFD": "fed_funds",
                                " USAVGEXPQ": "cons_exp"})
    raw = raw.reset_index(drop=True)
    raw['yc_slope'] = raw["ust_10yr"].astype(float)-raw['fed_funds'].astype(float)
    raw = raw.drop(["ust_10yr", "fed_funds"], axis = 1)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index('date')
    raw = raw.dropna()
    print(raw)
    return

feature_prep()