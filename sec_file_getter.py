import time

from secedgar.filings import Filing, FilingType

list_of_symbols = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO',
                   'JPM',
                   'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']


def get_files(symbols):
    for item in symbols:
        symb_filing = Filing(cik_lookup=item, filing_type=FilingType.FILING_10K, count=20,
                             user_agent='joeykess@umich.edu')
        symb_filing.save('assets/edgar/')
        print('Saved %s' % item)
        time.sleep(1)


if __name__ == "__main__":
    get_files(list_of_symbols)
