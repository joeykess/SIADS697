import edgar
import pandas as pd
import requests


def main():
    # edgar.download_index('test/', 2018, skip_all_present_except_last=False)
    selectedcompany = 'Alphabet Inc.'
    selectedreport = '10-Q'

    csv = pd.read_csv('test/2019-QTR4.tsv', sep='\t', lineterminator='\n', names=None)
    csv.columns.values[0] = 'Item'
    companyreport = csv[(csv['Item'].str.contains(selectedcompany)) & (csv['Item'].str.contains(selectedreport))]
    Filing = companyreport['Item'].str.split('|')
    Filing = Filing.to_list()

    for item in Filing[0]:

        if 'html' in item:
            report = item
    header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    url = 'https://www.sec.gov/Archives/' + report
    r = requests.get(url, headers=header)
    df = pd.read_html(r.text)
    document_index = df[0]
    document_index = document_index.dropna()

    document_name = document_index[document_index['Description'].str.contains(selectedreport)]
    document_name = document_name['Document'].str.split(' ')
    document_name = document_name[0][0]

    report_formatted = report.replace('-', '').replace('index.html', '')
    url = 'https://www.sec.gov/Archives/' + report_formatted + '/' + document_name
    print(url)
    r = requests.get(url, headers=header)

    df = pd.read_html(r.text)

    # for item in df:
    #     BS = (item[0].str.contains('Retained') | item[0].str.contains('Total Assets'))
    #     if BS.any():
    #         Balance_Sheet = item
    #
    # Balance_Sheet = Balance_Sheet.iloc[2:, [0, 2, 6]]
    #
    # header = Balance_Sheet.iloc[0]
    # Balance_Sheet = Balance_Sheet[1:]
    #
    # Balance_Sheet.columns = header
    #
    # Balance_Sheet.columns.values[0] = 'Item'
    # Balance_Sheet = Balance_Sheet[Balance_Sheet['Item'].notna()]
    #
    # Balance_Sheet[Balance_Sheet.columns[1:]] = Balance_Sheet[Balance_Sheet.columns[1:]].astype(str)
    # Balance_Sheet[Balance_Sheet.columns[1]] = Balance_Sheet[Balance_Sheet.columns[1]].map(lambda x: x.replace('(', '-'))
    # Balance_Sheet[Balance_Sheet.columns[2]] = Balance_Sheet[Balance_Sheet.columns[2]].map(lambda x: x.replace('(', '-'))
    #
    # Balance_Sheet[Balance_Sheet.columns[1]] = Balance_Sheet[Balance_Sheet.columns[1]].map(lambda x: x.replace(',', ''))
    # Balance_Sheet[Balance_Sheet.columns[2]] = Balance_Sheet[Balance_Sheet.columns[2]].map(lambda x: x.replace(',', ''))
    #
    # Balance_Sheet[Balance_Sheet.columns[1]] = Balance_Sheet[Balance_Sheet.columns[1]].map(lambda x: x.replace('—', '0'))
    # Balance_Sheet[Balance_Sheet.columns[2]] = Balance_Sheet[Balance_Sheet.columns[2]].map(lambda x: x.replace('—', '0'))
    #
    # Balance_Sheet[Balance_Sheet.columns[1:]] = Balance_Sheet[Balance_Sheet.columns[1:]].astype(float)
    # return Balance_Sheet
    return df


if __name__ == '__main__':
    print(main())
