# Title     : TODO
# Objective : TODO
# Created by: joeyk
# Created on: 6/7/2021

library(dplyr);
library(finreportr);

DJI_stocks <- list("AXP", 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO',
                   'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW')

for (symb in DJI_stocks){
  tryCatch(
    expr = {
      df1 <- GetIncome(symb, 2017)
      df2 <- GetBalanceSheet(symb, 2017)
      df3 <- GetCashFlow(symb, 2017)
      df <- bind_rows(df1, df2, df3)
      path_to_save <- sprintf('C:\\Users\\joeyk\\OneDrive\\PycharmProjects\\SIADS697\\assets\\edgar\\%s\\%s.csv', symb, symb)
      write.csv(df, path_to_save, row.names=FALSE)
      print(symb)
    },
    error =function(e){
      message(sprintf('Caught an error on symbol ', symb))
    }
  )
}