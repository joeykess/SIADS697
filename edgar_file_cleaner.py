import os
import json
import re

symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM',
           'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']
tags = ["us-gaap:LongTermDebtNoncurrent",
        "us-gaap:AssetsCurrent",
        "us-gaap:LiabilitiesCurrent",
        "us-gaap:NetIncomeLoss",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:OperatingIncomeLoss",
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
        "us-gaap:StockholdersEquity",
        "us-gaap:LongTermDebtCurrent"
        ]


def edgar_cleaner(list_of_symbols, list_of_tags):
    for symbol in list_of_symbols:
        dirs = os.getcwd() + '/assets/edgar/' + symbol + '/10-k/'
        for file in os.listdir(dirs):
            try:
                with open(dirs + file, 'r') as f:
                    contents = f.read()
                ans_dict = {}
                for item in tags:
                    ans = re.findall(r"<{}([\s\S]*?)>([\d]+)<\/{}>".format(item, item), contents)
                    ans_dict[item[8:]] = {item[0]: item[1] for item in ans}
            except:
                continue
        data_dict = {}
        for k, v in ans_dict.items():
            temp_dict = {}
            for k1, v1 in v.items():
                key = re.findall(r'contextRef=\"([A-Z0-9a-z]+)[\_\"].*', k1)[0]
                val = int(v1)
                temp_dict[key] = val
            data_dict[k] = temp_dict
        with open(os.getcwd() + '/assets/edgar/' + symbol + '/' + symbol + '_10k.txt', 'w') as outfile:
            json.dump(data_dict, outfile)
        print('Wrote to ', symbol)


if __name__ == "__main__":
    edgar_cleaner(symbols, tags)
