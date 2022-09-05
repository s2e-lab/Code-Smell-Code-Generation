def getQuotes(self, symbol, start, end):
        """
        Get historical prices for the given ticker symbol.
        Date format is 'YYYY-MM-DD'

        Returns a nested list.
        """
        try:
            start = str(start).replace('-', '')
            end = str(end).replace('-', '')

            url = 'http://ichart.yahoo.com/table.csv?s=%s&' % symbol + \
                'd=%s&' % str(int(end[4:6]) - 1) + \
                'e=%s&' % str(int(end[6:8])) + \
                'f=%s&' % str(int(end[0:4])) + \
                'g=d&' + \
                'a=%s&' % str(int(start[4:6]) - 1) + \
                'b=%s&' % str(int(start[6:8])) + \
                'c=%s&' % str(int(start[0:4])) + \
                'ignore=.csv'
            days = urllib.urlopen(url).readlines()
            values = [day[:-2].split(',') for day in days]
            # sample values:[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Clos'], \
            #              ['2009-12-31', '112.77', '112.80', '111.39', '111.44', '90637900', '109.7']...]
            data = []
            for value in values[1:]:
                data.append(Quote(value[0], value[1], value[2], value[3], value[4], value[5], value[6]))

            dateValues = sorted(data, key = lambda q: q.time)
            return dateValues

        except IOError:
            raise UfException(Errors.NETWORK_ERROR, "Can't connect to Yahoo server")
        except BaseException:
            raise UfException(Errors.UNKNOWN_ERROR, "Unknown Error in YahooFinance.getHistoricalPrices %s" % traceback.format_exc())