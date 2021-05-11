from sec_edgar_downloader import Downloader

dl = Downloader(r"assets")
lis = ['BA', 'IBM', 'AAPL', 'GOOG']
for x in lis:
    dl.get("10-K", x)
