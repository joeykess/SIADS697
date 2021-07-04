#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import re


def main():

    df = pd.read_csv('fundamentals_spy.csv', index_col=0)
    inst_ric = df['Instrument'].unique()
    inst_ric = {k:k.split('.')[0] for k in inst_ric}
    inst_ric['BRKb.N'] = 'BRK-B'
    inst_ric['BFb.N'] = 'BF-B'
    df['Instrument'] = df.replace({"Instrument": inst_ric})


if __name__ == '__main__':
    main()
