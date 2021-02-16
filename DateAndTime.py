from SetUp import *
print('loading DT')

class DateAndTime():

    def __init__(self, from_date='20190403', until_date='20210131'):

        self.from_date = from_date
        self.until_date = until_date


        time_fmt = "%H:%M"
        opening_hours_str, closing_hours_str = "07:00", "15:30"  # to be adjusted depending on summerwinter time
        self.opening_hours = datetime.datetime.strptime(opening_hours_str, time_fmt).time()
        self.closing_hours = datetime.datetime.strptime(closing_hours_str, time_fmt).time()

        self.day_count = ql.Business252()
        self.cal = ql.Germany()

        #all dates
        dates_list = pd.date_range(self.from_date, self.until_date, freq='D')
        dates_list.freq = None
        self.dates_list = [elt.strftime('%Y%m%d') for elt in dates_list if (pd.Timestamp(elt).date().strftime("%A") not in ['Saturday', 'Sunday'])]

        #expi dates
        self.first_matu = pd.Timestamp(self.from_date).date()
        w = self.first_matu.weekday()
        # Friday is weekday 4
        if w >= 4:
            self.first_matu = self.first_matu + datetime.timedelta(days=7-(w-4))
        else:
            self.first_matu = self.first_matu + datetime.timedelta(days=4-w)
        self.last_matu = (pd.Timestamp(until_date) + pd.DateOffset(years=2)).date().strftime('%Y-%m-%d')
        dates_expi = list(pd.date_range(self.first_matu, self.last_matu, freq='W'))
        # dates_expi = [elt - datetime.timedelta(2) for elt in dates_expi]
        dates_expi = [datetime.datetime.combine(elt, self.closing_hours) for elt in dates_expi if elt.day in [15, 16, 17, 18, 19, 20, 21]]
        self.dates_expi = [self.get_last_working(elt) for elt in dates_expi]
        self.dates_expi_trim = [elt for elt in self.dates_expi if elt.month in [3, 6, 9, 12]]
        self.dates_expi_sem = [elt for elt in self.dates_expi if elt.month in [6, 12]]


    def time_between(self, a, b):
        d1 = a.date()
        d2 = b.date()
        d1 = ql.Date(a.day, a.month, a.year)
        d2 = ql.Date(b.day, b.month, b.year)
        nbd = self.cal.businessDaysBetween(d1, d2)
        TimeA = datetime.datetime.combine(datetime.date.today(), a.time())
        TimeB = datetime.datetime.combine(datetime.date.today(), b.time())
        if TimeB > TimeA:
            addhours = (TimeB - TimeA).total_seconds() / 3600
        else:
            addhours = -((TimeA - TimeB).total_seconds() / 3600)
        # return (nbd + addhours/8.5)/252
        return (nbd + addhours / 12.5) / 252  # so that the night counts for 4 hours
        # we neglect the 1 hour imprecision at the time of switch from winter to summer daylight saving


    def get_last_working(self, dt):
        dt_dt = dt.date()
        dt_ql = ql.Date(dt_dt.day, dt_dt.month, dt_dt.year)
        while self.cal.isHoliday(dt_ql):
            dt_ql = dt_ql - 1
        return (dt_ql.to_date())

    def get_matu_list(self, reference_date):
        ts = pd.Timestamp(reference_date)
        dates_expi_M = [elt for elt in self.dates_expi if (elt > ts) and (elt < ts + pd.Timedelta(31 * 4 + 6, unit='D'))]
        dates_expi_T = [elt for elt in self.dates_expi_trim if
                        (elt >= ts + pd.Timedelta(31 * 4 + 6, unit='D')) and (elt < ts + pd.DateOffset(months=13))]
        dates_expi_L = [elt for elt in self.dates_expi_sem if (elt >= ts + pd.DateOffset(months=13))]
        return ([elt.strftime('%Y%m%d') for elt in dates_expi_M + dates_expi_T + dates_expi_L])
