from SetUp import *
from DateAndTime import DateAndTime
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm


class Pricing():

    def __init__(self):
        # super(Pricing, self).__init__()
        # self.DT = DT
        self.smile_sliding_coef = 1
        self.moneyness_range = (-0.6, 0.6) #for 1 year, in sqrt(T)


    def american_vanilla_pricer(self, S, K, d1, d2, r, repo, sigma, N=200, greek=''):

        sigma = sigma / 100
        option_type = ql.Option.Call
        risk_free_rate = 0.00

        ql.Settings.instance().evaluationDate = d1
        payoff = ql.PlainVanillaPayoff(option_type, K)
        am_exercise = ql.AmericanExercise(d1, d2)
        american_option = ql.VanillaOption(payoff, am_exercise)

        # eu_exercise = ql.EuropeanExercise(d2)
        # european_option = ql.VanillaOption(payoff, eu_exercise)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))

        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(d1, risk_free_rate, self.DT.day_count))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(d1, repo, self.DT.day_count))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(d1, self.DT.cal, sigma, self.DT.day_count))
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", N)
        american_option.setPricingEngine(binomial_engine)

        if greek == '':
            return american_option.NPV()

        elif greek in ['vega', 'delta-vega']:
            price = american_option.NPV()
            american_option_shock = ql.VanillaOption(payoff, am_exercise)
            flat_vol_ts_shock = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(d1, self.DT.cal, sigma+0.01, self.DT.day_count))
            bsm_process_shock = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts_shock)
            binomial_engine_shock = ql.BinomialVanillaEngine(bsm_process_shock, "crr", N)
            american_option_shock.setPricingEngine(binomial_engine_shock)
            if greek == 'vega':
                return price, american_option_shock.NPV() - price
            elif greek == 'delta-vega':
                return price, american_option.delta(), american_option_shock.NPV() - price

        elif greek == 'delta':
            return american_option.NPV(), american_option.delta()


    def european_vanilla_pricer(self, S, K, d1, d2, r, repo, sigma, type, greek=''):

        sigma = sigma / 100

        if type == 'Call':
            option_type = ql.Option.Call
        else:
            option_type = ql.Option.Put

        risk_free_rate = 0.00

        ql.Settings.instance().evaluationDate = d1

        payoff = ql.PlainVanillaPayoff(option_type, K)


        # am_exercise = ql.AmericanExercise(d2, d2)
        # american_option = ql.VanillaOption(payoff, am_exercise)

        eu_exercise = ql.EuropeanExercise(d2)
        european_option = ql.VanillaOption(payoff, eu_exercise)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))

        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(d1, risk_free_rate, self.DT.day_count)
        )
        dividend_yield = ql.YieldTermStructureHandle(
            ql.FlatForward(d1, repo, self.DT.day_count)
        )
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(d1, self.DT.cal, sigma, self.DT.day_count))

        # vol_quote = ql.SimpleQuote(sigma)

        bsm_process = ql.BlackScholesMertonProcess(spot_handle,
                                                   dividend_yield,
                                                   flat_ts, flat_vol_ts)
        simple_engine = ql.AnalyticEuropeanEngine(bsm_process)
        european_option.setPricingEngine(simple_engine)

        if greek == '':
            return european_option.NPV()
        elif greek == 'vega':
            return european_option.NPV(), european_option.vega()/100
        elif greek == 'delta':
            return european_option.NPV(), european_option.delta()
        elif greek == 'delta-vega':
            return european_option.NPV(), european_option.delta(), european_option.vega()/100


    def vanilla_pricer(self, S, K, r, sigma, fwdRatio, type, exerc=1, greek=''):
        if type=='1':
            type = "Call"
        elif type=='0':
            type = "Put"

        T = self.DT.cal.businessDaysBetween(self.d1, self.d2) / 252.0

        indic = abs(1 - fwdRatio) / max(1.0 / 12.0, T) ** 0.5
        # not taking moneyness into account because all calls must have same method + only OTM calls anyway
        repo = - math.log(fwdRatio) / max(1 / 128, T)

        if (type == "Call") and (indic > 0.002) and (exerc != 0):

            return self.american_vanilla_pricer(S, K, self.d1, self.d2, r, repo, sigma, 200, greek)

        # elif (type == "Call")  and (self.udl not in indexlist) and (exerc != 0):
        #
        #     american = self.american_vanilla_pricer(S, K, self.d1, self.d2, r, repo, sigma, 200, greek)
        #     european = self.european_vanilla_pricer(S, K, self.d1, self.d2, r, repo, sigma, type, greek)
        #
        #     if isinstance(american, tuple):
        #         american_p = american[0]
        #         european_p = european[0]
        #     else:
        #         american_p = american
        #         european_p = european
        #     if abs(american_p-european_p)/S*10000 > 5:
        #         print('!!!!!!!!!!!!!!!!!!!!!!!!!! {} ; {} ; {} ; {} ; {} ; {} ; {} ; {}   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(abs(american_p-european_p)/S*10000, american_p, european_p, S, K, T, sigma, fwdRatio))
        #     return american
        else:
            return self.european_vanilla_pricer(S, K, self.d1, self.d2, r, repo, sigma, type, greek)


    def pcal1(self, opt, field=''):
        r = 0
        moneyness = math.log(opt.StrikePrice / (opt.FVU * self.FwdRatio))

        if field == 'bid':
            sigma = float(min(200, max(10, self.vol_spline_bid(moneyness))))
        else:
            sigma = float(min(200, max(10, self.vol_spline_ask(moneyness))))

        price, vega = self.vanilla_pricer(opt.FVU, opt.StrikePrice, r, sigma, self.FwdRatio, opt.PutOrCall, exerc=opt.ExerciseStyle, greek='vega')

        if vega==0:
            return sigma, price, vega, moneyness
        else:
            shiftsigma = -(price - opt[field]) / vega
            shiftsigma = min(max(-sigma*0.2, shiftsigma), sigma*0.2) #capfloor to the move
            return sigma + shiftsigma, price, vega, moneyness


    def pcal2(self, opt, field=''):
        r = 0
        if field == 'bid':
            sigma = float(min(200, max(10, self.vol_spline_bid(opt.moneyness))))
        else:
            sigma = float(min(200, max(10, self.vol_spline_ask(opt.moneyness))))

        price = self.vanilla_pricer(opt.FVU, opt.StrikePrice, r, sigma, self.FwdRatio, opt.PutOrCall, opt.ExerciseStyle, greek = '')

        return sigma, price


    def pcal3(self, opt, field=''):
        r = 0
        price = self.vanilla_pricer(opt.FVU*1.01, opt.StrikePrice, r, opt[field+'_iv'], self.FwdRatio, opt.PutOrCall, opt.ExerciseStyle, greek = '')
        return price - opt[field+'_model_price']


    def pcal4(self, opt):
        r = 0
        self.d1 = opt.d1
        self.d2 = opt.d2
        sigma = (opt.iv_bid + opt.iv_ask)/2
        price, delta, vega = self.vanilla_pricer(opt.FVU, opt.StrikePrice, r, sigma, opt.FwdRatio, opt.PutOrCall, opt.ExerciseStyle, greek='delta-vega')
        return delta, vega


    def pcal5(self, opt, sigma):
        r = 0
        self.d1 = opt.d1
        self.d2 = opt.d2
        price = self.vanilla_pricer(opt.FVU, opt.StrikePrice, r, sigma, opt.FwdRatio, opt.PutOrCall, opt.ExerciseStyle, greek='')
        return price


    def pcal6(self, opt, field=''):
        r = 0
        self.d1 = opt.name.date()
        self.d2 = pd.Timestamp(opt.matu).date()
        self.d1 = ql.Date(self.d1.day, self.d1.month, self.d1.year)
        self.d2 = ql.Date(self.d2.day, self.d2.month, self.d2.year)
        price = self.vanilla_pricer(opt.FVU, opt.StrikePrice, r, opt[field+'_iv'], opt.FwdRatio, opt.PutOrCall, opt.ExerciseStyle, greek='')
        return price



class FittingSpline(Pricing):

    def __init__(self, udl, DT, folder1, folder2):
        self.DT = DT
        super(FittingSpline, self).__init__()
        self.udl = udl
        self.folder1 = folder1
        self.folder2 = folder2

        self.df_all = pd.read_pickle(self.folder1 + '/Quotes_' + self.udl + '.pkl')

        #filter quotes
        self.df_all = self.df_all.loc[(self.df_all.bid!=0) & (self.df_all.ask!=0)]
        self.df_all.dropna(subset=['bid', 'ask'], inplace=True)

        self.df_udl = self.df_all.loc[self.df_all.matu == 'UDL'].copy()
        self.df_udl['FVU'] = (self.df_udl.bid + self.df_udl.ask)/2

        self.descr_cols = ['PutOrCall', 'StrikePrice', 'ContractMultiplier', 'ExerciseStyle']
        self.floor_spread_iv = 0.5
        self.max_error = 8 #in bps

        try:
            self.df_params = pd.read_pickle(self.folder2 + '/Params_' + self.udl + '.pkl')
        except:
            mi = pd.MultiIndex(levels=[[], []],
                          codes=[[], []],
                          names=['ts', 'matu'])
            self.df_params = pd.DataFrame(index=mi, columns=['spline_bid', 'spline_ask', 'FwdRatio', 'Spot', 'Error', 'Fwd_computed'])


    def fit_all(self):

        done_already = [elt.strftime('%Y%m%d') for elt in set([elt.date() for elt in self.df_params.index.get_level_values(0)])]

        for reference_date in [elt for elt in self.DT.dates_list if (elt not in done_already)]:  #['20190606']
            print(reference_date)
            matulist = [elt for elt in self.DT.get_matu_list(reference_date) if elt != reference_date]
            for matu in matulist:
                print('   ' + matu)
                self.ini_day(reference_date, matu)
                self.fit_day()

            self.df_params.to_pickle(self.folder2 + '/Params_' + self.udl + '.pkl')


    def ini_day(self, present_date, maturity_date):
        try:
            dfp = self.df_params.loc[pd.IndexSlice[:, maturity_date], :]
        except:
            dfp = pd.DataFrame()

        if dfp.shape[0] == 0:
            self.vol_spline_bid = lambda x: 30.0
            self.vol_spline_ask = lambda x: 30.0
            self.FwdRatio = 1.0
        else:
            maxts = dfp.index.get_level_values(0).max()
            params = dfp.loc[maxts, maturity_date]
            self.vol_spline_bid = params.spline_bid
            self.vol_spline_ask = params.spline_ask
            self.FwdRatio = params.FwdRatio

        self.present_date = pd.Timestamp(present_date)
        self.present_date_dt = self.present_date.date()
        self.maturity_date_str = maturity_date
        self.maturity_date = pd.Timestamp(maturity_date)
        self.maturity_date_dt = self.maturity_date.date()

        self.d1 = ql.Date(self.present_date_dt.day, self.present_date_dt.month, self.present_date_dt.year)
        self.d2 = ql.Date(self.maturity_date.day, self.maturity_date.month, self.maturity_date.year)


        self.df = self.df_all.loc[self.df_all.matu == maturity_date]
        self.df = self.df.loc[(self.df.index >= np.datetime64(self.present_date)) & (self.df.index < np.datetime64(self.present_date+pd.DateOffset(days=1)))]
        self.df = pd.merge(self.df_udl[['FVU']], self.df, how='inner', left_index=True, right_index=True)

        if self.df.shape[0] > 0:
            self.FVU = self.df.FVU[0]
            # self.T = time_between(self.present_date, self.maturity_date)
            self.T = self.DT.cal.businessDaysBetween(self.d1, self.d2) / 252.0
            self.df['moneyness_T'] = self.df.apply(lambda opt: math.log(opt.StrikePrice / (opt.FVU * 0.98))/(max(3.0/12.0, self.T)**0.5), axis='columns')
            self.df = self.df.loc[(self.df.moneyness_T > self.moneyness_range[0]) & (self.df.moneyness_T < self.moneyness_range[1])]


    def fit_day(self):
        time_pace = list(set(self.df.index.tolist()))
        time_pace = [elt for elt in time_pace if elt.to_pydatetime().time().minute in [elt*5 for elt in range(12)]]
        time_pace.sort()
        self.df_time = pd.DataFrame(columns=self.descr_cols)

        for self.time_slice in time_pace:
            self.get_new_vol_params(self.time_slice)
            if self.slice_success:
                self.df_params.loc[(self.time_slice, self.maturity_date_str), :] = self.vol_spline_bid, self.vol_spline_ask, self.FwdRatio, self.FVU, self.error, self.Fwd_computed
            # print(self.iter)



    def get_new_vol_params(self, time_slice):
        keep_cols = [elt for elt in self.df_time.columns if elt not in ['FVU', 'bid', 'ask', 'matu', 'T', 'moneyness_T']]
        self.df_lo = self.df.loc[time_slice].dropna()

        if (type(self.df_lo) == pd.core.frame.DataFrame) and (self.df_lo.shape[0] >= 5):
            #eliminate if only calls and on low strikes or only puts  onhigh strikes
            FVU = self.df_lo.FVU.iloc[0]
            eliminateK = [K for K, PC in zip(self.df_lo.StrikePrice, self.df_lo.PutOrCall) if (self.df_lo.StrikePrice.tolist().count(K)==1) and (((K<FVU) and (PC=='1')) or ((K>FVU) and (PC=='0')))]
            self.df_lo = self.df_lo.loc[~self.df_lo.StrikePrice.isin(eliminateK)]

        if (type(self.df_lo) == pd.core.frame.DataFrame) and (self.df_lo.shape[0] >= 5):
            self.slice_success = True
            self.Fwd_computed = False

            self.df_time = pd.merge(self.df_lo.sort_values(by='StrikePrice', ascending=True), self.df_time[keep_cols], how='left', left_on=self.descr_cols, right_on=self.descr_cols)
            for elt in ['sensidelta_ask']: #, 'bid_vega', 'ask_vega']:
                if elt not in self.df_time.columns:
                    self.df_time[elt] = 0.9*self.FVU/100
                else:
                    self.df_time[elt].fillna(0.9*self.FVU/100, inplace=True)  #100% is max for delta which will limit impact
            self.df_time[['bid_iv', 'bid_model_price', 'bid_vega', 'moneyness']] = self.df_time.apply(lambda x: self.pcal1(x, 'bid'), axis=1, result_type='expand')
            self.df_time[['ask_iv', 'ask_model_price', 'ask_vega', 'moneyness']] = self.df_time.apply(lambda x: self.pcal1(x, 'ask'), axis=1, result_type='expand')
            self.df_time.dropna(subset=['bid_iv', 'ask_iv'], inplace=True)

        if (type(self.df_lo) == pd.core.frame.DataFrame) and (self.df_lo.shape[0] >= 5) and (type(self.df_time) == pd.core.frame.DataFrame) and (self.df_time.shape[0] >= 5):
                with warnings.catch_warnings():
                    # self.df_time.sort_values(by=['moneyness'], ascending=[True], inplace=True)
                    # warnings.simplefilter("ignore")
                    warnings.filterwarnings("ignore")
                    W = 1/(self.df_time.ask_iv - self.df_time.bid_iv).apply(lambda x: max(x, self.floor_spread_iv)) * (1-abs(self.df_time.sensidelta_ask/self.FVU*100))**2
                    # we want 1/w[i] to be around std(Y[i]) in order to be consistent with s = len(W)
                    # see : https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
                    #if we consider std(Y[i]) = 1 we want Wmax around 1
                    X = self.df_time.moneyness
                    Y = self.df_time.bid_iv
                    Wbid = self.df_time.bid_vega * W
                    Wbid = Wbid / max(Wbid)

                    fitok = False
                    leeway = 1
                    sp_s = self.vol_spline_bid
                    while (not fitok) and (leeway<=8):
                        self.vol_spline_bid = UnivariateSpline(X, Y, Wbid, k=2, s=len(Wbid)*leeway)
                        fitok = ~np.isnan(float(self.vol_spline_bid(0)))
                        leeway = leeway * 2
                        if leeway>2:
                            print('    leeway : ' + str(leeway))
                    if not fitok:
                        self.vol_spline_bid = sp_s

                    Y = self.df_time.ask_iv
                    Wask = self.df_time.ask_vega * W
                    Wask = Wask / max(Wask)

                    fitok = False
                    leeway = 1
                    sp_s = self.vol_spline_ask
                    while (not fitok) and (leeway<=8):
                        self.vol_spline_ask = UnivariateSpline(X, Y, Wask, k=2, s=len(Wask)*leeway)
                        fitok = ~np.isnan(float(self.vol_spline_ask(0)))
                        leeway = leeway * 2
                        if leeway > 2:
                            print('    leeway : ' + str(leeway))
                if not fitok:
                    self.vol_spline_ask = sp_s

                self.df_time[['bid_iv', 'bid_model_price']] = self.df_time.apply(lambda x: self.pcal2(x, 'bid'), axis=1, result_type='expand')
                self.df_time[['ask_iv', 'ask_model_price']] = self.df_time.apply(lambda x: self.pcal2(x, 'ask'), axis=1, result_type='expand')

                df_time_itm = self.df_time.loc[((self.df_time.moneyness_T > -0.1) & (self.df_time.PutOrCall == '1')) | (
                            (self.df_time.moneyness_T <= 0.1) & (self.df_time.PutOrCall == '0'))]
                self.error = abs((df_time_itm.bid_model_price + df_time_itm.ask_model_price) / 2 - (
                            df_time_itm.bid + df_time_itm.ask) / 2).mean() / FVU * 10000

                # self.df_time['mid'] = (self.df_time.bid + self.df_time.ask) / 2
                # self.df_time['mid_fv'] = (self.df_time.bid_model_price + self.df_time.ask_model_price) / 2
                # self.df_time['error'] = abs(self.df_time['mid'] - self.df_time['mid_fv']) / FVU * 10000

                if (self.error > self.max_error * max(1, self.T)) or (time_slice.time().minute == 5) or (0.9*self.FVU/100 in self.df_time.sensidelta_ask):
                    self.df_time['sensidelta_bid'] = self.df_time.apply(lambda x: self.pcal3(x, 'bid'), axis='columns')
                    self.df_time['sensidelta_ask'] = self.df_time.apply(lambda x: self.pcal3(x, 'ask'), axis='columns')
                    # find best FwdRatio adjustment with WLS
                    W = 1 / (self.df_time.ask_iv - self.df_time.bid_iv).apply(lambda x: max(x, self.floor_spread_iv)**2)
                    Wbid = abs(self.df_time.sensidelta_bid) * W
                    Wask = abs(self.df_time.sensidelta_ask) * W

                    X = np.float64(np.concatenate((np.array(self.df_time.sensidelta_bid), np.array(self.df_time.sensidelta_ask))))
                    Y = np.float64(np.concatenate((np.array(self.df_time.bid - self.df_time.bid_model_price), np.array(self.df_time.ask - self.df_time.ask_model_price))))
                    W = np.float64(np.concatenate((np.array(Wbid), np.array(Wask))))
                    wls_model = sm.WLS(Y, X, weights=W)
                    self.results = wls_model.fit()
                    self.FwdRatio = self.FwdRatio * (1 + self.results.params[0] * 0.01)
                    # print('Fwd updated')
                    self.Fwd_computed = True
        else:
            self.slice_success = False


    def graph(self, day, matu):

        self.df_params = pd.read_pickle(self.folder2 + '/Params_' + self.udl + '.pkl')
        self.ini_day(day, matu)
        self.df_params_matu = self.df_params.xs(matu, level=1, drop_level=True)
        self.df_params_matu = self.df_params_matu.loc[self.df_params_matu.Error < 20]
        self.df_params_matu['day'] = self.df_params_matu.index
        self.df_params_matu['day'] = self.df_params_matu.day.apply(lambda x: x.date())
        self.df_params_matu = self.df_params_matu.loc[self.df_params_matu.day == pd.Timestamp(day).date()]

        selected_slices = sorted(list(set(self.df_params_matu.index)))[24::24]  # [120::120]
        self.df_graph = pd.merge(self.df.loc[selected_slices], self.df_params_matu, how='left', left_index=True, right_index=True)
        self.df_graph.dropna(subset=['spline_bid', 'spline_ask'], inplace=True)
        selected_slices = [elt for elt in selected_slices if elt in self.df_graph.index]

        self.df_graph['moneyness'] = self.df_graph.apply(lambda opt: math.log(opt.StrikePrice / (opt.FVU * opt.FwdRatio)), axis='columns') # math.log(opt.StrikePrice / opt.FVU * opt.FwdRatio)

        self.df_graph['bid_iv'] = self.df_graph.apply(lambda opt: float(opt.spline_bid(opt.moneyness)), axis='columns')
        self.df_graph['ask_iv'] = self.df_graph.apply(lambda opt: float(opt.spline_ask(opt.moneyness)), axis='columns')

        self.df_graph['fv_bid'] = self.df_graph.apply(lambda opt: self.pcal6(opt, 'bid'), axis='columns')
        self.df_graph['fv_ask'] = self.df_graph.apply(lambda opt: self.pcal6(opt, 'ask'), axis='columns')

        self.df_graph['time_slice'] = self.df_graph.index

        self.df_graph_pt1 = self.df_graph.pivot_table(index=['time_slice', 'moneyness'], values=['bid_iv', 'ask_iv'])
        ncols = 1
        nrows = math.ceil(len(selected_slices)/ncols)

        f, a = plt.subplots(nrows, ncols, squeeze=False, figsize=(10, 10))
        for i, time_slice in enumerate(selected_slices):
            self.df_graph_pt1.xs(time_slice).plot(ax=a[int(i/ncols), i % ncols], title=time_slice)
            a[int(i/ncols), i % ncols].get_legend().remove()
            a[int(i/ncols), i % ncols].set_title(time_slice.to_pydatetime().strftime("%H:%M:%S"), fontsize=6)
            a[int(i/ncols), i % ncols].set_xlabel('')

        f.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12, hspace = 1)  # create some space below the plots by increasing the bottom-value
        a.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncols)
        plt.show()

        print('')
        print('')

        ncols = 2
        nrows = len(selected_slices)

        fields = ['bid', 'fv_ask', 'ask']
        for elt in fields:
            self.df_graph[elt] = self.df_graph[elt]-self.df_graph['fv_bid']
        self.df_graph['fv_bid'] = 0

        f, a = plt.subplots(nrows, ncols, squeeze=False, figsize=(10, 10))
        self.df_graph_pt2 = self.df_graph.pivot_table(index=['time_slice', 'PutOrCall', 'StrikePrice'], values=['fv_bid'] + fields)
        for i, time_slice in enumerate(selected_slices):
            self.df_graph_pt2.xs(time_slice).xs('0').plot(ax=a[i, 0], fontsize=6)
            a[i, 0].get_legend().remove()
            a[i, 0].set_title(time_slice.to_pydatetime().strftime("%H:%M:%S") + " " + "Puts", fontsize=6)
            a[i, 0].set_xlabel('')
            self.df_graph_pt2.xs(time_slice).xs('1').plot(ax=a[i, 1], fontsize=6)
            a[i, 1].get_legend().remove()
            a[i, 1].set_title(time_slice.to_pydatetime().strftime("%H:%M:%S") + " " + "Calls", fontsize=6)
            a[i, 1].set_xlabel('')

        f.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12, hspace=1)  # create some space below the plots by increasing the bottom-value
        a.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncols)
        plt.show()

if __name__ == '__main__':
    udl = 'DAI'
    reference_date = '20210105'
    folder1 = 'D:/Users/GitHub/TradesDynamics/processed'
    folder2 = 'D:/Users/GitHub/TradesDynamics/parameters'
    DT = DateAndTime(reference_date, reference_date)
    FS = FittingSpline(udl, DT, folder1, folder2)
    FS.fit_all()
    # FS.graph("20190403", "20190920")
