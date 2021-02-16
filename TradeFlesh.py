from SetUp import *
from PricingAndCalibration import Pricing
from Clustering import Clustering
from matplotlib.lines import Line2D
from DateAndTime import DateAndTime

class TradeFlesh(Pricing):

    def __init__(self, udl, DT, folder1, folder2):
        self.DT = DT
        super(TradeFlesh, self).__init__()

        self.udl = udl
        self.folder1 = folder1
        self.folder2 = folder2

        self.max_error = 20


    def pct_aggressivity(self):

        self.df_trades = pd.read_pickle(self.folder1 + '/Trades_' + self.udl + '.pkl')

        self.df_trades = self.df_trades.sort_index()
        self.df_trades['dtf'] = pd.to_datetime(self.df_trades.index)
        self.df_trades['dtf_d'] = self.df_trades['dtf'].apply(lambda x: x.date())

        self.df_params = pd.read_pickle(self.folder2 + '/Params_' + self.udl + '.pkl')

        self.df_params = self.df_params.loc[self.df_params.Error < self.max_error]

        #get udl price at the time of trade
        for reference_date in self.DT.dates_list:
            try:
                self.df_udl = pd.read_pickle(self.folder1 + '/raw/Quotes_' + '{}_{}.pkl'.format(self.udl, reference_date))
                self.df_udl = self.df_udl.loc[self.df_udl.matu == 'UDL']
                self.df_udl_bid = self.df_udl.loc[self.df_udl.bidask == 'bid']
                self.df_udl_ask = self.df_udl.loc[self.df_udl.bidask == 'ask']
                newindex = self.df_trades.loc[self.df_trades.dtf_d == pd.Timestamp(reference_date).date()].index
                self.df_udl_bid = self.df_udl_bid.reindex(newindex, method='ffill')
                self.df_udl_ask = self.df_udl_ask.reindex(newindex, method='ffill')
                self.df_trades.loc[self.df_trades.dtf_d == pd.Timestamp(reference_date).date(), 'FVU'] = (self.df_udl_bid.level.values + self.df_udl_ask.level.values)/2
            except:
                print('missing raw file : Quotes_' + '{}_{}.pkl'.format(self.udl, reference_date))

        for elt in ['spline_bid', 'spline_ask', 'FwdRatio', 'Spot']:
            self.df_trades[elt] = None

        #put params in df_trades
        for matu in set(self.df_trades.matu):
            try:
                self.df_params_matu = self.df_params.xs(matu, level=1, drop_level=True)
                self.df_params_matu['calib_ts'] = self.df_params_matu.index
                newindex = self.df_trades.loc[self.df_trades.matu == matu].index

                if self.df_params_matu.shape[0] > 0:
                    self.df_params_matu = self.df_params_matu.reindex(newindex, method='ffill')  #if we take 'nearest', it is not consistant with building an indicator later
                    self.df_trades.loc[self.df_trades.matu == matu, ['spline_bid', 'spline_ask', 'FwdRatio', 'Spot', 'calib_ts']] = self.df_params_matu[['spline_bid', 'spline_ask', 'FwdRatio', 'Spot', 'calib_ts']]
            except:
                print('no parameter for :' + matu)

        self.df_trades['d1'] = self.df_trades.dtf_d.apply(lambda x: ql.Date(x.day, x.month, x.year))
        self.df_trades['d2'] = self.df_trades.matu.apply(lambda x: pd.Timestamp(x).date())
        self.df_trades['d2'] = self.df_trades.d2.apply(lambda x: ql.Date(x.day, x.month, x.year))
        self.df_trades['T'] = self.df_trades.apply(lambda opt: self.DT.cal.businessDaysBetween(opt.d1, opt.d2) / 252.0, axis='columns')

        self.df_trades['pricable'] = ~self.df_trades.calib_ts.isnull()

        #filter if time_to_calib is more than 5 mins
        dfsub = self.df_trades[self.df_trades.pricable]
        dfsub['calib_ts'] = abs((dfsub.index - dfsub.calib_ts.values).astype('timedelta64[s]'))
        self.df_trades.loc[self.df_trades.pricable, 'pricable'] = dfsub.apply(lambda x: (x.calib_ts < 60*5) and (x.bidentry != pd.Timestamp('1970-01-01 00:00:00')) and (x.askentry != pd.Timestamp('1970-01-01 00:00:00')), axis=1)

        # eliminate far OTM options
        self.df_trades.loc[self.df_trades.pricable, 'moneyness'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: math.log(opt.StrikePrice / (opt.FVU * opt.FwdRatio)), axis='columns')
        self.df_trades.loc[self.df_trades.pricable, 'moneyness_T'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: opt.moneyness/(max(3.0/12.0, opt['T'])**0.5), axis='columns')
        self.df_trades.loc[self.df_trades.pricable, 'pricable'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: (opt.moneyness_T > self.moneyness_range[0]) and (opt.moneyness_T < self.moneyness_range[1]), axis='columns')

        self.df_trades.loc[self.df_trades.pricable, 'iv_bid'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: opt.spline_bid(opt.moneyness), axis=1)
        self.df_trades.loc[self.df_trades.pricable, 'iv_ask'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: opt.spline_ask(opt.moneyness), axis=1)
        self.df_trades.loc[self.df_trades.pricable, 'pricable'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: (opt.iv_bid>5) and (opt.iv_ask<200) and (opt.iv_bid<opt.iv_ask), axis='columns')

        self.df_trades.loc[self.df_trades.pricable, 'theo_bid'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: self.pcal5(opt, opt.iv_bid), axis=1)

        self.df_trades.loc[self.df_trades.pricable, 'theo_ask'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: self.pcal5(opt, opt.iv_ask), axis=1)
        self.df_trades.loc[self.df_trades.pricable, 'aggressivity'] = self.df_trades.loc[self.df_trades.pricable].apply(lambda opt: self.get_aggressivity(opt), axis=1)

        self.df_trades['pricable'] = self.df_trades.pricable & (~self.df_trades.aggressivity.isnull())
        self.df_trades.drop(['d1', 'd2'], axis=1, inplace=True) #cannot be pickled as it is a quantlib object
        self.df_trades.to_pickle(self.folder2 + '/FleshedTrades_' + self.udl + '.pkl')


    def get_aggressivity(self, opt):
        mid = (opt.theo_bid + opt.theo_ask)/2
        half_spread = (opt.theo_ask - opt.theo_bid)/2
        if half_spread <= 0:
            return np.nan
        else:
            return min(1, max(-1, (opt.px - mid) / half_spread))


    def graph_aggressivity(self, gdate, highlight_cluster=[]):

        self.df_trades = pd.read_pickle(self.folder2 + '/FleshedTrades_' + self.udl + '.pkl')
        self.df_trades['sz'] = [64 if elt in highlight_cluster else 8 for elt in range(self.df_trades.shape[0])]

        dtf = pd.to_datetime(gdate).date()
        self.df_trades = self.df_trades.loc[self.df_trades.dtf_d == dtf]

        matu_list = sorted(list(set(self.df_trades.matu)))
        matu_list = [elt for elt in matu_list if (pd.Timestamp(elt).date().month in [3, 6, 9, 12]) and ((pd.Timestamp(elt) - pd.Timestamp(gdate)).total_seconds() < 60 * 60 * 24 * 365)]
        ncols = 1
        nrows = len(matu_list)

        self.df_graph = self.df_trades[['matu', 'px', 'bid', 'ask', 'theo_bid', 'theo_ask', 'dtf', 'sz']].copy()
        self.df_graph.columns = ['matu', 'ExecPrc', 'bid', 'ask', 'theo_bid', 'theo_ask', 'dtf', 'sz']

        fields = ['ExecPrc', 'ask', 'bid', 'theo_ask']
        self.df_graph['theo_bid_'] = self.df_graph['theo_bid']
        self.df_graph.theo_bid_.fillna(self.df_graph.bid, inplace=True)
        for elt in fields:
            self.df_graph[elt] = self.df_graph[elt] - self.df_graph['theo_bid_']
        self.df_graph['theo_bid'] = self.df_graph['theo_bid'] - self.df_graph['theo_bid']


        f, a = plt.subplots(nrows, ncols, squeeze=False, figsize=(10, 10))  #, figsize=(10, 10)

        for i, matu in enumerate(matu_list):
            dfm = self.df_graph.loc[self.df_graph.matu == matu]
            # dfm['dtf_t'] = dfm.dtf.apply(lambda x: x.time())
            x = [elt for elt in dfm['dtf']]
            a[i, 0].scatter(x, dfm['theo_bid'], color='red', s=8)
            a[i, 0].scatter(x, dfm['theo_ask'], color='red', s=8)
            a[i, 0].scatter(x, dfm['ExecPrc'], color='black', marker='x', s=dfm.sz.values)
            if i > 0:
                a[0, 0].get_shared_x_axes().join(a[0, 0], a[i, 0])

            ye = np.concatenate(([[0]*dfm.shape[0]], [(dfm.ask-dfm.bid).values]), axis=0)
            a[i, 0].errorbar(x, dfm['bid'].values, yerr=ye, fmt='none')
            # a[i, 0].get_legend().remove()
            a[i, 0].set_title('maturity : ' + matu, fontsize=8)
            a[i, 0].set_xlabel('', fontsize=6)
            a[i, 0].set_ylabel('', fontsize=6)

        # f.subplots_adjust(top=3, left=0.1, right=3, bottom=0.5, hspace=0.4)  # create some space below the plots by increasing the bottom-value
        colors = ['black', 'blue', 'red']
        shapes = {'black': 'x', 'blue': '', 'red': 'o'}
        linest = {'black': 'None', 'blue': '-', 'red': 'None'}
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle=linest[c], marker=shapes[c]) for c in colors]
        labels = ['Trading Price', 'bid-ask spread', 'model bid&ask']
        a.flatten()[-1].legend(lines, labels, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        plt.gcf().autofmt_xdate()
        plt.show()


    def get_intensity(self):
        self.df_trades = pd.read_pickle(self.folder2 + '/FleshedTrades_' + self.udl + '.pkl')
        self.df_trades['d1'] = self.df_trades.dtf_d.apply(lambda x: ql.Date(x.day, x.month, x.year))
        self.df_trades['d2'] = self.df_trades.matu.apply(lambda x: pd.Timestamp(x).date())
        self.df_trades['d2'] = self.df_trades.d2.apply(lambda x: ql.Date(x.day, x.month, x.year))

        dft = self.df_trades.loc[self.df_trades.pricable].apply(self.pcal4, axis=1, result_type='expand')
        self.df_trades.loc[self.df_trades.pricable, 'delta'] = dft.iloc[:, 0]
        self.df_trades.loc[self.df_trades.pricable, 'vega'] = dft.iloc[:, 1]

        self.df_trades['sega'] = self.df_trades.vega * self.df_trades.moneyness

        if isinstance(self.df_trades.qty[0], str):
            self.df_trades.volume = self.df_trades.qty.apply(lambda x: float(x))

        self.df_trades['vega_intensity'] = self.df_trades.volume * self.df_trades.ContractMultiplier * self.df_trades.vega * self.df_trades.aggressivity
        self.df_trades['delta_intensity'] = self.df_trades.volume * self.df_trades.ContractMultiplier * self.df_trades.delta * self.df_trades.FVU * self.df_trades.aggressivity

        self.df_trades.drop(['d1', 'd2'], axis=1, inplace=True)  # cannot be pickled as it is a quantlib object
        self.df_trades.to_pickle(self.folder2 + '/FleshedTrades_' + self.udl + '.pkl')


    def graph_sensitivity(self, field, day):

        param = {'vega': 'iv', 'delta': 'FwdRatio'}
        self.df_trades = pd.read_pickle(self.folder2 + '/FleshedTrades_' + self.udl + '.pkl')

        self.df_params = pd.read_pickle(self.folder2 + '/Params_' + self.udl + '.pkl')
        self.df_params = self.df_params.loc[self.df_params.Error < self.max_error]

        #select day
        self.df_params['day'] = self.df_params.index
        self.df_params['day'] = self.df_params.day.apply(lambda x: x[0].date())
        self.df_params = self.df_params.loc[self.df_params.day == pd.Timestamp(day).date()]
        #reduce maturities

        matu_list = sorted(list(set([elt[1] for elt in self.df_params.index])))
        matu_list = [elt for elt in matu_list if (pd.Timestamp(elt).date().month in [3,6,9,12]) and ((pd.Timestamp(elt) - pd.Timestamp(day)).total_seconds() < 60*60*24*365)]

        ncols = 1
        nrows = len(matu_list)

        fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(10, 10))
        # ax3 = np.zeros(shape=(nrows,ncols))
        # rspine = np.zeros(shape=(nrows, ncols))
        for i, matu in enumerate(matu_list):
            self.df_params_matu = self.df_params.xs(matu, level=1, drop_level=True)


            if self.df_params_matu.shape[0]>0:
                #we get the iv on the first udl spot of the morning
                MFV = self.df_params_matu.Spot[0]
                self.df_params_matu['moneyness'] = self.df_params_matu.Spot.apply(lambda x: math.log(MFV / x))
                self.df_params_matu['iv'] = self.df_params_matu.apply(lambda x: (x.spline_bid(x.moneyness) + x.spline_ask(x.moneyness))/2, axis=1)

                self.df_trades = self.df_trades.loc[self.df_trades.pricable]
                self.df_trades_matu = self.df_trades.loc[self.df_trades.matu == matu, [field + '_intensity']]

                #merge trades and params dataframes
                self.df_trades_matu.index = self.df_trades_matu.index.round('min')
                dfg = self.df_trades_matu.groupby(self.df_trades_matu.index).agg({field + '_intensity': sum})
                dfg = dfg.reindex(self.df_params_matu.index, method=None)
                dfg[param[field]] = self.df_params_matu[param[field]]

                dfg[field + '_intensity_ewma'] = dfg[field + '_intensity'].fillna(0).ewm(halflife=30).mean()
                dfg[param[field] + '_ewma'] = dfg[param[field]].fillna(0).ewm(halflife=30).mean()


                ax[i, 0].spines['bottom'].set_position('zero')
                ax3 = ax[i, 0].twinx()
                rspine = ax3.spines['right']
                ax3.set_frame_on(True)
                ax3.patch.set_visible(False)
                fig.subplots_adjust(right=0.7)

                ax[i, 0].bar(dfg.index.values, dfg[field + '_intensity'].values, width=0.001, color='green')
                ax[i, 0].plot(dfg.index.values, dfg[field + '_intensity_ewma'].values, color='green')
                ax[i, 0].set_title('maturity ' + matu, fontsize=8)
                ax3.plot(dfg.index, dfg[param[field]], color='DarkBlue')
                ax3.plot(dfg.index, dfg[param[field]+'_ewma'], color='DarkBlue')
                if i==0:
                    legend=ax3.legend([ax[i, 0].get_lines()[0], ax3.get_lines()[0]], \
                       [field + ' intensity', param[field]], bbox_to_anchor=(1, 1), fontsize=12)
        plt.show()


if __name__ == '__main__':
    udl = 'DAI'
    reference_date = '20210105'
    folder1 = 'D:/Users/GitHub/TradesDynamics/processed'
    folder2 = 'D:/Users/GitHub/TradesDynamics/parameters'
    DT = DateAndTime(reference_date, reference_date)
    TF = TradeFlesh(udl, DT, folder1, folder2)
    TF.pct_aggressivity()
    TF.graph_aggressivity('20190710')
    TF.get_intensity()
    TF.graph_sensitivity('vega', '20190710')

    C = Clustering(udl)
    C.prepare_data()
    TF.graph_aggressivity('20190710', C.trades(1))
