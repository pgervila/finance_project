# scientific modules
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


# plot modules
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
# import seaborn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

# import built-in, text acquisition and processing libraries
import os
import glob
from bs4 import BeautifulSoup
import requests  # instead of urllib
import talib as ta
import datetime


class Data_Input_Output(object):
    
    def __init__(self):
        
        """ It initializes path- and label-related attributes needed to define
            multiple URLs from where data is downloaded and updated.
            It also defines empty containers for data storage """
        
        self.path_saved_data = '/Users/PG/Paolo/python_repos/finance_proj/data'
        file = 'indices_labels.csv'
        labels = pd.read_csv(os.path.join(self.path_saved_data, file))
        self.tickers = labels['COUNTRY_INDECES'][1:].tolist()
        self.index_names = labels['index_name'][1:].tolist()
        self.FX_list = labels['currency'][1:].tolist()
        self.FX_EUR_names = labels['FX_EUR'][1:].tolist()
        self.FX_USD_names = labels['FX_USD'][1:].tolist()
        self.data = []
        self.data_upd = []
        self.EMA_lengths = [10, 20, 50, 100, 200]
    
    def load_ALL_saved_data(self):
        """ It loads indices saved in HDF format into panda's DFs"""
        ## save current working directory
        cwd = os.getcwd()
        ## change to directory where indices data are saved
        os.chdir(self.path_saved_data) 
        ## get most recent HDF file
        newest_HDF_file = max(glob.iglob('*.h5'),
                              key=os.path.getctime) 
        ## import with hdf-pandas
        list_df = []
        for idx,_ in enumerate(self.index_names):
            df_imported = pd.read_hdf(newest_HDF_file,
                                      key = 'df_' + str(idx))
            list_df.append(df_imported)            
        self.data = list_df
        ## back to working directory
        os.chdir(cwd) 
               
    def get_upd_data(self, i, sec_type, FX):
        """ Downloads up-to-date data for a given index or currency i"""
        if sec_type == 'indices':           
            names = self.index_names        
        elif sec_type == 'currencies':            
            if FX == 'EUR':            
                names = self.FX_EUR_names
            elif FX == 'USD':
                names = self.FX_USD_names
        web_site = ('http://www.investing.com/' +
                    sec_type + '/' +
                    str(names[i]) + '-historical-data')
        ## get site content and make soup
        headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/601.4.4 (KHTML, like Gecko) Version/9.0.3 Safari/601.4.4"}
        r = requests.get(web_site, headers = headers)
        ## SOL 1 : use pd.read_html
        soup = BeautifulSoup(r.content, "lxml")
        upd_prices = pd.read_html(
            str(soup.find('table',
                          attrs = {'class':
                                   'genTbl closedTbl historicalTbl'}
                          )))[0]
        upd_prices['Date'] = pd.to_datetime(upd_prices['Date'].str.replace(",", ""),
                                            format = "%b %d %Y")
        upd_prices.set_index('Date', inplace = True)
        upd_prices.sort_index(inplace = True)
        if sec_type == 'indices':
            upd_prices.drop(['Vol.','Change %'], axis = 1, inplace = True)
            upd_prices['ticker'] = self.tickers[i]
            upd_prices.columns = ["close","open","high","low","ticker"]
        elif sec_type == 'currencies':
            if FX == 'EUR':
                upd_prices = pd.DataFrame(1/upd_prices['Price'])
                upd_prices.columns = ["1/EUR"]
            elif FX == 'USD':
                upd_prices = pd.DataFrame(1/upd_prices['Price'])
                upd_prices.columns = ["1/USD"]
        return upd_prices
    
     
     
    def assemble_upd_data(self, i):
        """ It merges saved data with updated one"""
        ## find list position of first index in EUR

        if self.FX_list[i] != 'euro' :        
        
            upd_prices = self.get_upd_data(i, 'indices', [])
            upd_FX_EUR = self.get_upd_data(i, 'currencies', 'EUR')
            upd_data = pd.concat([upd_prices, upd_FX_EUR], axis=1).dropna() 
            ## add columns
            upd_data['close_EUR'] = upd_data.close*upd_data['1/EUR']
            upd_data['open_EUR'] = upd_data.open*upd_data['1/EUR']
            upd_data['high_EUR'] = upd_data.high*upd_data['1/EUR']
            upd_data['low_EUR'] = upd_data.low*upd_data['1/EUR'] 
                                
            if self.FX_list[i] == 'local':
                
                upd_FX_USD = self.get_upd_data(i,'currencies','USD')
                
                upd_data = pd.concat([upd_data, upd_FX_USD], axis=1).dropna() 
             
                upd_data['close_USD'] = upd_data.close*upd_data['1/USD']
                upd_data['open_USD'] = upd_data.open*upd_data['1/USD']
                upd_data['high_USD'] = upd_data.high*upd_data['1/USD']
                upd_data['low_USD'] = upd_data.low*upd_data['1/USD']
                
                ##merge with old data
                upd_data = upd_data.combine_first(self.data[i])
                
            else:  ## this step is actually not needed
            
                upd_data['1/USD'] = 1
                upd_data['close_USD'] = upd_data.close
                upd_data['open_USD'] = upd_data.open
                upd_data['high_USD'] = upd_data.high
                upd_data['low_USD'] = upd_data.low
                
                upd_data = upd_data.combine_first(self.data[i])
        else:
            
            upd_data = self.get_upd_data(i, 'indices', [])
            upd_data = upd_data.combine_first(self.data[i])
        
        self.data_upd = upd_data
        return upd_data


    def save_upd_data(self):
        """ It saves update data to a new HDF file"""
        # store in hdf-pandas
        ## save current working directory
        cwd = os.getcwd()
        ## change to directory where indices data are saved
        os.chdir(self.path_saved_data)
        ##today_date
        today_date = str(datetime.datetime.today().date())
        with pd.HDFStore('DF_ALL_INDICES_' + today_date + '.h5','w') as store_DFs:
            for idx,_ in enumerate(self.data):
                print(idx)
                store_DFs.put('df_' + str(idx),
                              self.assemble_upd_data(idx))
        ## close store
##        store_DFs.close()
        ## back to working directory
        os.chdir(cwd)


class Compute_indicators(Data_Input_Output): ##TODO : define alternat class to load separate index not in list !!!

    def load_index(self,i):
        ## save current working directory
        cwd = os.getcwd()
        ## change to directory where indices data are saved
        os.chdir(self.path_saved_data) 
        ## get most recent HDF file
        newest_HDF_file = max(glob.iglob('*.h5'),
                                 key=os.path.getctime) 
        ## import with hdf-pandas          
        self.index_data = pd.read_hdf(newest_HDF_file,
                                key = 'df_' + str(i))
        ## back to working directory
        os.chdir(cwd) 
        

    def comp_volatility(self,i):
        """ Computes alternative volatility and adds column to index DF"""
                    
        pp = 20  ## num of days for volat EMA average    
        if self.FX_list[i] == 'local':

            ret_for_volat = (self.index_data['low_EUR']/
                             self.index_data['close_EUR'].shift(1) - 1)
            ret_for_volat.loc[(ret_for_volat > 0.)] = 0.
            ret_for_volat = ret_for_volat.abs()
            volat_scale_f = 100*np.sqrt(252) ## factor to annualize volatilities
            self.index_data['volat_LC'] = volat_scale_f*pd.ewma(
                ret_for_volat,
                min_periods = pp,
                span = pp
                )
            self.index_data['ret_for_volat_LC'] = ret_for_volat              
        else:
            ret_for_volat = (self.index_data['low']/
                             self.index_data['close'].shift(1) - 1)
            ret_for_volat.loc[(ret_for_volat > 0.)] = 0.
            ret_for_volat = ret_for_volat.abs()
            volat_scale_f = 100 * np.sqrt(252) ## factor to annualize volatilities
            self.index_data['volat_LC'] = volat_scale_f*pd.ewma(
                ret_for_volat,
                min_periods = pp,
                span = pp
                )
            self.index_data['ret_for_volat_LC'] = ret_for_volat
        
        return self.index_data['volat_LC']

    def simulate_future_volat():
        ## simulate with pre-defined prob distrib how volat will evolve in future n days
        ## p = [p1,p2,p3] sum(p) = 1
        pass


    def comp_EMA_volat(self):
        EMA_volat = {}
        [EMA_volat.update({'EMA_volat_' + str(n): pd.ewma(
            self.index_data.volat_LC,
            min_periods = n,
            span = n).values}) for n in self.EMA_lengths]
        EMA_volat = pd.DataFrame(EMA_volat, index = self.index_data.index)
        self.index_data= self.index_data.join(EMA_volat)

    def get_volat_slope(self, n, EMA_days):
        """ n is # days for shift"""
        
        self.index_data['volat_slope'] = self.index_data['EMA_volat_' +
                                                         str(EMA_days)].diff(n)        
        self.index_data['volat_accel'] = self.index_data['volat_slope'].diff(n)
        self.index_data['volat_accel_delta'] = self.index_data['volat_accel'].diff(n)

    def comp_EMA_index(self, i):      
        EMA_index = {}     
        if self.FX_list[i] == 'local':
            [EMA_index.update({'EMA_' + str(length):pd.ewma(
                self.index_data.close_EUR,
                min_periods = length,
                span = length).values}) for length in self.EMA_lengths]
            EMA_index = pd.DataFrame(EMA_index,
                                     index = self.index_data.index)
            self.index_data = self.index_data.join(EMA_index)

        else:
            [EMA_index.update({'EMA_' + str(length):pd.ewma(
                self.index_data.close,
                min_periods = length,
                span = length).values}) for length in self.EMA_lengths]
            EMA_index = pd.DataFrame(
                EMA_index,
                index = self.index_data.index
                )
            self.index_data = self.index_data.join(EMA_index)

    def comp_volat_EMA_volat_ratio(self, i):
        volat_EMA_volat_index = {}
        [volat_EMA_volat_index.update({
            'volat_EMA_volat_' + str(length):
            getattr(self.index_data, 'volat_LC')/
            getattr(self.index_data, 'EMA_volat_' + str(length))
            }) for length in self.EMA_lengths ]
        volat_EMA_volat_index = pd.DataFrame(volat_EMA_volat_index,
                                             index = self.index_data.index)
        self.index_data = self.index_data.join(volat_EMA_volat_index)

    def comp_stoch_osc(self, i):
        if self.FX_list[i] == 'local':
            close = getattr(self.index_data, 'close' + '_EUR')
            high = getattr(self.index_data, 'high' + '_EUR')
            low = getattr(self.index_data, 'low' + '_EUR')
        else:
            close = getattr(self.index_data, 'close')
            high = getattr(self.index_data, 'high')
            low = getattr(self.index_data, 'low')
            
        stoch_K = 100 * (close - pd.rolling_min(low, window = 14)) / (
            pd.rolling_max(high, window = 14) - pd.rolling_min(low, window = 14)
            )
        stoch_K =pd.rolling_mean( stoch_K, window = 3)           
        stoch_D = pd.rolling_mean( stoch_K, window = 3)
        STOCH_ = pd.concat([stoch_K, stoch_D], axis = 1)
        STOCH_.columns = ['SK','SD']
        self.index_data['OSC_SLOW'] = STOCH_['SD']
            
#        stoch_K, stoch_D = ta.STOCH( high, low, close, 14)
#        STOCH_ = pd.DataFrame({'STOCH_K':stoch_K, 'STOCH_D':stoch_D }, index = self.index_data.)
        self.stoch_osc = STOCH_
        return self.stoch_osc
        
    def comp_RSI_osc(self, i):
        if self.FX_list[i] == 'local':            
            close = (getattr(self.index_data, 'close' + '_EUR').
                     fillna(method ='ffill').values)
        else:
            close = (getattr(self.index_data,'close').
                     fillna(method = 'ffill').values)
        
        self.index_data['RSI'] = pd.DataFrame({'RSI':ta.RSI(close)},
                                    index = self.index_data.index )

    def get_segs_data(self, indicator, price, threshold = 1.): ## use it with volat as indicator as well !!!
        """ Finds length, location and basic stats of segments above and below
        a given threshold.
        ######################################
        indicator : time series e.g. ratio of
        volat to EMA_volat, 50-100-200 days
        price : time series of prices
        threshold : scalar
        ######################################
        """

##        need to pass i !!

##        indicator =
##        if self.FX_list[i] == 'local':
##            price =
##        else:
##            price = 

        df = pd.concat([indicator,price] , axis = 1).dropna()
        df.columns = ['indicator','price']
        df['sign'] = np.sign((df['indicator'] - threshold))
        grouped = df.groupby(
            (df['sign'] != df['sign'].shift()).cumsum()
            )
        ## custom functions to aggregate
        def group_return(arr):
            return arr[-1]/arr[0] - 1
        def group_max_return(arr):
            return arr.max()/arr[0] - 1
        def group_min_return(arr):
            return arr.min()/arr[0] - 1
        
        segments_data = pd.DataFrame({
            'start': df.index[[ list_[0]
                                for list_ in
                                grouped['sign'].grouper.indices.values()]],
            'dur': grouped['sign'].sum(),
            'max_seg': grouped['indicator'].max(),
            'max_idx': grouped['indicator'].idxmax(),
            'min_seg': grouped['indicator'].min(),
            'min_idx': grouped['indicator'].idxmin(),
            'mean_seg': grouped['indicator'].mean(),
            'cum_ret': grouped['price'].aggregate(group_return),
            'max_cum_ret': grouped['price'].aggregate(group_max_return),
            'min_cum_ret': grouped['price'].aggregate(group_min_return)
            })
        
        return segments_data


    def track_volat_cycle(self, indicator, price, volat, threshold = 1.):
        ''' tracks index returns starting from stats-based expected maximum of volat/EMA_volat curve ratio
        and selects highest return until new upwards EMAvolat-crossing event occurs
    
        Parameters
        ==========    
        -indicator: df or ts. Ratio of a price-based or price-return-based series
         to an EMA of the very same series    
        -price: df or array, index prices    
        -volat: df or array, volatility of index prices        
        -threshold: scalar. Value to which ratio parameter compares in order to define segs    
        '''
        
        ## some calculations before starting
        segs_info = self.get_segs_data(indicator, price, threshold)
        max_segs_mean = segs_info[segs_info.dur > 0].max_seg.mean()
        mean_volat = np.mean(volat) ## compute long term volatility mean
        ## better pd.rolling_mean(volat,100)

        ## define DF on which to operate, get rid of nan values
        df = pd.concat([indicator, price, volat, self.index_data.OSC_SLOW,
                        self.index_data.RSI, pd.rolling_mean(volat, 100),
                        pd.rolling_std(volat, 100)] ,
                       axis = 1).dropna()
        # define columns names
        df.columns = ['indicator','price','volat',
                      'STOCH_SLOW','RSI','roll_mean_volat',
                      'roll_std_volat']
        ## define segs based on threshold
        df['segm'] = np.where(df.indicator>threshold, True, False)
        #associate DIFFERENT groups to segs
        df['groups'] = (df['segm'] != df['segm'].shift()).cumsum()
        
        ## define potential start before triggering
        ## start point must have been in posit segment for at least two steps
        df['start'] = (df['segm']) & (df['segm'].shift(2))
        ## trigger conditions ( may be verified multiple times for a given seg)
        df['trigger'] = ((df['start']) &
                         (df['indicator'] < df['indicator'].shift()) &
                         (df['indicator'] >= max_segs_mean) &
                         (df['volat'] > df['roll_mean_volat'] + df['roll_std_volat']) &
                         (df['STOCH_SLOW'] < 20) &
                         (df['RSI'] < 40))
        ## sum number of potential trigger points along each positive segments
        df['trigg_cumsum'] = df['trigger'].groupby((df['segm'] !=
                                                    df['segm'].shift()).cumsum()).cumsum()

        ## confirm cycle start with trigger
        ## pick only first trigger for a given posit segment
        df['trigger'] = (df['trigger']) & (df['trigg_cumsum'] == 1)
        
        ## define all potential stops( indicator crossing upwards)
        df['stops'] = (df['segm']) & (df['segm'].shift() == False)

        ##define new DF based on groups and triggers
        ##Goal is to define lag indicator showing if trigger is present in previous seg
        grouped_segs = df['trigger'].groupby(df['groups'])
        ## find out if trigger is present in group True/False
        df_2 = pd.DataFrame(grouped_segs.sum())
        df_2.columns = ['trigger_in_group']
        df_2 = df_2.reset_index()
        df_2.columns = ['groups','trigger_in_group']
        ## define column that tells if trigger was present in previous pos segment
        df_2['lag_1'] = df_2.trigger_in_group.shift(2)
        ## merge based on groups
        df = df.reset_index().merge(df_2).set_index('index')
        
        ## now define TRUE stops using lagged indicator
        df['stops'] = (df['lag_1'] == 1) & (df['stops'])
        ## aggregate all true trigger + stop points into single column
        df['trigg_stop_points'] = df['trigger'] + df['stops']
        
        ## define groups based on TRUE trigger-stop points
        df['groups_trigg_stop_points'] = df['trigg_stop_points'].cumsum()
        
        ## keep only relevant groups ( odd groups, since start == 0 is never in volat cycle )
        df['period'] = df['groups_trigg_stop_points'] % 2 != 0
        df = df[df['period']]

        ##define customized aggreg functions
        def group_return(arr):
            return arr[-1]/arr[0] - 1
        def group_max_return(arr):
            return arr.max()/arr[0] - 1
        def group_min_return(arr):
            return arr.min()/arr[0] - 1

        price_grouped = df['price'].groupby(df['groups_trigg_stop_points'])
        
        volat_cycle_data = pd.DataFrame({
        'cycle_ret': price_grouped.aggregate(group_return),
        'cycle_max_ret': price_grouped.aggregate(group_max_return),
        'cycle_idx_max_ret': price_grouped.idxmax(),
        'cycle_min_ret': price_grouped.aggregate(group_min_return),
        'cycle_idx_min_ret': price_grouped.idxmin(),
        'dur': df.groupby(df['groups_trigg_stop_points']).size(),
        'trigger_idx': df[df['trigger']].index})

        return volat_cycle_data


    def find_roll_present_past_segs_length(self, indicator, threshold = 1.):
        """ Finds rolling length of segments where indicator is above or below
        a given threshold and adds them to df together with corresponding
        segment group. It also creates df containing lengths of current and
        past segments also by groups.
        It finally merges both DFs
        ####################################################
        indicator : e.g. ratio of volat to EMA_volat, 50-100-200 days
        threshold : level to define segments
        """
        df = pd.DataFrame(indicator.dropna())
        df['sign'] = np.sign((df - threshold))
        grouped_segs = df['sign'].groupby((df['sign'] != df['sign'].shift()).cumsum())
        df_count_in_segs = grouped_segs.cumsum()
        df['counts'] = df_count_in_segs ## Take abs values ????
        ## add id for different groups
        df['groups'] = (df['sign'] != df['sign'].shift()).cumsum()

        df_2 = pd.DataFrame(grouped_segs.sum())
        df_2.columns = ['tot_days']
        df_2 = df_2.reset_index()
        df_2.columns = ['groups','tot_days']
        df_2['lag_1'] = df_2.tot_days.shift()
        df_2['lag_2'] = df_2.tot_days.shift(2)
        df_2['lag_3'] = df_2.tot_days.shift(3)
        df_2['lag_4'] = df_2.tot_days.shift(4)
        df_2 = df_2.set_index('groups')
        
        self.rolling_lengths = pd.merge(df, df_2,
                                        left_on = 'groups',
                                        right_index = True)

        return self.rolling_lengths


    def get_max_min_fut_return_in_period(self, i, ret_period):
        """ given a prices TS with price(t) it finds best, worst
            and actual return occured within a future period = ret_period from t

        INPUT

        i : index number. integer

        ret_period : period in days. integer

        """

        if self.FX_list[i] == 'local':        
            close_price = 'close_EUR'
        else:          
            close_price =  'close'
        ## define empty frame
        df_fut_returns = pd.DataFrame()
        df_fut_returns['prices'] = self.index_data[close_price]
        df_fut_returns['actual_ret_in_period'] = (df_fut_returns['prices'].
                                    pct_change(ret_period).
                                    shift(-ret_period))

        ## compute rets for i-periods and shift back
        for i in range(1, ret_period):
            df_fut_returns['ret_' + str(i)] = (
                1 + df_fut_returns['prices'].pct_change(i)
                ).shift(-i)
        ## select ret columns and pick max for each row 
        RET_cols = df_fut_returns.filter(regex = "ret_")
        df_fut_returns['max_ret_in_period'] = RET_cols.max(axis =  1,
                                                           skipna = False)
        df_fut_returns['min_ret_in_period'] = RET_cols.min(axis = 1,
                                                           skipna = False)
        ## find which day in future max occurs for ech day
        df_fut_returns['num_days_max_ret'] = RET_cols.idxmax(axis = 1,
                                                             skipna = False)
        df_fut_returns['num_days_max_ret'] = pd.to_numeric(df_fut_returns['num_days_max_ret'].
                                                           replace(regex = True,
                                                                   to_replace = r'\D',
                                                                   value = r'')
                                                           )
        df_fut_returns['num_days_min_ret'] = RET_cols.idxmin(axis = 1,skipna = False)
        df_fut_returns['num_days_min_ret'] = pd.to_numeric(df_fut_returns['num_days_min_ret'].
                                                           replace(regex = True,
                                                                   to_replace = r'\D',
                                                                   value = r'')
                                                           )
        ## keep only relevant cols
        self.max_min_fut_returns = df_fut_returns.filter(regex = r"(prices|max_ret|min_ret)")
        
        return self.max_min_fut_returns

    def get_price_volat_pattern_OLD(self, indicator, price,
                                    threshold, trig_day, crossing,
                                    length_cross, up_down):        
        ''' show patterns in index prices after volatility has been above
        or below EMA-volat for days == trig_day
        
        Parameters:
        ##############        
        -indicator: numpy array. ratio of a series to one of its EMA,
        e.g. ratio of volatility to EMA_100 of volatility
        -price: numpy array. Series of prices for chosen security 
        -threshold: scalar. Level to which crossing for indicator value occurs
        -trig_day: integer. Number of days past the upward crossing
        from which returns start to be computed
        -crossing : string. 'Yes' in order to compute returns including not only 'up' days
        but also extra days after crossing downwards
        -length_cross: integer. Number of days to be considered for return calculation
        after downwards crossing
        -up_down: string. 'up' for return calculation over segments
        where indicator > 1, 'down' otherwise
        
        Output:
        ###############        
        '''    
        start = -1
        price_volat_pattern = []
        count = 0        
        ##define indicator    
        if (up_down == 'up'):
            for idx,x in enumerate(indicator):                
                if start < 0 and x > threshold :
                    start = idx                    
                elif start >= 0 and (idx - start) <= trig_day and x <= threshold:
                    start = - 1                    
                elif start >= 0 and (idx - start) == trig_day and x > threshold:
                    p_start = price[idx]
                    trigger = idx                    
                elif start >= 0 and x <= threshold:                    
                    if crossing == 'yes':                        
                        if idx + length_cross < len(price -1):
                            dur = idx + length_cross -start
                            p_end = price[idx + length_cross]
                            ret = 100 * (p_end/p_start - 1)
                            count += 1
                            price_volat_pattern.append([ indicator.index[start].date(),
                                                         dur,
                                                         ret,
                                                         count])
                            start = -1
                        else:
                            dur = len(price) -start
                            p_end = price[-1]
                            ret = 100 * (p_end/p_start - 1)
                            count += 1
                            price_volat_pattern.append([indicator.index[start].date(),
                                                        dur,
                                                        ret,
                                                        count])
                            start = -1                            
                    else:
                        p_end = price[idx]
                        dur = idx -trigger
                        ret = 100 * (p_end/p_start - 1)
                        count += 1
                        price_volat_pattern.append([indicator.index[start].date(),
                                                    dur,
                                                    ret,
                                                    count])
                        start = -1    
        price_volat_pattern = np.array(price_volat_pattern)
        price_volat_pattern = pd.DataFrame({'trigger_idx': price_volat_pattern[:,0],
                                           'dur': price_volat_pattern[:,1],
                                           'cycle_ret': price_volat_pattern[:,2],
                                           'counts': price_volat_pattern[:,3]})
        return price_volat_pattern

    def fun_concat(indicator, price, volat):
        df = pd.concat([indicator, price, volat],axis = 1).dropna()
        df.columns = ['indicator','price','volat']
        return df

## MODIF : pandas based
    def get_price_volat_pattern(self, i, indicator, price, trig_day,
                                length_cross, threshold = 1.,
                                crossing='yes', up_down='up'):
        
        ''' show patterns in index prices after volatility has been above
        or below EMA-volat for days == trig_day
        
        Parameters:
        ##############

        -trig_day: integer. Number of days past the upward crossing
        from which returns start to be computed
        -length_cross: integer. Number of days to be considered for return calculation
        after downwards crossing
        -threshold: scalar. Level to which crossing for indicator value occurs
        -crossing : string. 'Yes' in order to compute returns including not only 'up' days
        but also extra days after crossing downwards
        -up_down: string. 'up' for return calculation over segments
        where indicator > 1, 'down' otherwise
        
        Output:
        ###############
        DF with duration, returns, min and max returns for volat cycles defined by 
        
        '''

        ## define DF on which to operate, get rid of useless nan values
        if self.FX_list[i] == 'local':        
            close_price = 'close_EUR'            
        else:          
            close_price =  'close'
        df = pd.concat([indicator,close_price],axis=1).dropna().copy()
        # define columns names
        df.columns = ['indicator','price']
        # define segments sign
        df['sign'] = np.sign((df['indicator'] - threshold))
        ## define groups out of segs
        groups = (df['sign'] != df['sign'].shift()).cumsum()
        df['groups'] = groups       
        #group by segms groups
        grouped = df['sign'].groupby(groups)
        
        # rolling segs days
        df['segs_roll_days'] = grouped.cumsum()
        
        ## define trigger points
        df['trigger'] = (df['sign'] > 0) & (df['segs_roll_days'] >= trig_day)
        ## keep only starting points
        df['trigger'] = df['trigger'] & (df['trigger'] != df['trigger'].shift())


        ## define new df to track triggers in previous segs
        df_2 = pd.DataFrame()
        df_2 = df['trigger'].groupby(groups).sum()
        df_2.columns = ['trigger_in_group']
        df_2 = df_2.reset_index()
        df_2.columns = ['groups','trigger_in_group']
        ## define column that tells if trigger was present in previous segment
        df_2['lag_1'] = df_2.trigger_in_group.shift(1)
        df_2 = df_2.set_index('groups')

        ##merge DFs
        df = pd.merge(df, df_2, left_on = 'groups',right_index = True)

        ## get closing points
        df['stop'] = (df['sign'] < 0) & (df['lag_1']) & (df.segs_roll_days.abs() <= length_cross + 1)
        ##keep only last point
        df['stop'] = df['stop'] & (df['stop'] != df['stop'].shift(-1))

        ## keep all TRUE trigger-stop points
        df['true_cycle'] = df['trigger'] | df['stop']

        ##make groups with trigg-stop points
        df['groups_trigg_stop_points'] = df['true_cycle'].cumsum()

        ## keep only relevant groups ( odd groups, since start == 0 is never in volat cycle )
        df['period'] = df['groups_trigg_stop_points'] % 2 != 0
        df = df[df['period']]

        ##define customized aggreg functions
        def group_return(arr):
            return 100*(arr[-1]/arr[0] - 1)
        def group_max_return(arr):
            return 100*(arr.max()/arr[0] - 1)
        def group_min_return(arr):
            return 100*(arr.min()/arr[0] - 1)

        price_grouped = df['price'].groupby(df['groups_trigg_stop_points'])
        price_volat_pattern_data = pd.DataFrame({
        'cycle_ret': price_grouped.aggregate(group_return),
        'cycle_max_ret': price_grouped.aggregate(group_max_return),
        'cycle_min_ret': price_grouped.aggregate(group_min_return),
        'dur': df.groupby(df['groups_trigg_stop_points']).size(),
        'trigger_idx': df[df['trigger']].index})

        return price_volat_pattern_data


    def plot_volat_pattern(self, i, indicator, EMA_idx, plot_delta, max_days):
        """ EMA_idx is index to pick from list self.EMA_lengths = [20, 50, 100, 200] """

        volat_EMA = "volat_EMA_volat_" + str(self.EMA_lengths[EMA_idx])

        if self.FX_list[i] == 'local':        
            close_price = 'close_EUR'            
        else:          
            close_price =  'close'

##        prices = self.index_data[close_price].ix[self.EMA_lengths[EMA_idx]:]
##        indicator = self.index_data[volat_EMA].ix[self.EMA_lengths[EMA_idx]:]

        indicator = indicator[self.EMA_lengths[EMA_idx]:]
        price = self.index_data[close_price].ix[self.EMA_lengths[EMA_idx]:]
        
        self.lc_list = np.array(range(1,max_days,plot_delta))
        self.td_list = np.array(range(1,max_days,plot_delta))
##        self.X,self.Y = np.meshgrid(self.lc_list,self.td_list)
    
        plot_array = np.zeros((6, len(self.lc_list), len(self.td_list)))
        tot_years = len(self.index_data[close_price])/float(252)

        #ordinary loop better than list comprehension to produce multiple arrays
        for lc,c_days in enumerate(self.lc_list):
            for td,t_days in enumerate(self.td_list) :
                pattern = self.get_price_volat_pattern( i , indicator,
                                                        price, t_days, c_days)
                plot_array[0,td,lc] = pattern['cycle_ret'].mean()
                plot_array[1,td,lc] = pattern['cycle_max_ret'].mean()
                plot_array[2,td,lc] = pattern['cycle_min_ret'].mean()
                plot_array[3,td,lc] = pattern['cycle_ret'].std()
                plot_array[4,td,lc] = pattern['cycle_ret'].skew()
                plot_array[5,td,lc] = len(pattern)/tot_years 

        exp_returns = plot_array[0]
        exp_max_returns = plot_array[1]
        exp_min_returns = plot_array[2]
        exp_std = plot_array[3]
        exp_skew = plot_array[4]
        num_opport = plot_array[5]
       
        ##plot 
        fig, ax = plt.subplots(2, 3,  figsize=(7, 6))
        self.plot_subplot(ax, 0, 0, exp_returns,
                          False, 'expected_returns')
        self.plot_subplot(ax, 0, 1, exp_max_returns,
                          False, 'expected_max_returns')
        self.plot_subplot(ax, 0, 2, exp_min_returns,
                          False, 'expected_min_returns')
        self.plot_subplot(ax, 1, 0, num_opport,
                          True, "expected # occurences / year")
        self.plot_subplot(ax, 1, 1, exp_std,
                          True, 'expected standard deviation')
        self.plot_subplot(ax, 1, 2, exp_skew,
                          True, 'expected skewness')


    def plot_subplot(self, ax, i, j, data, set_x_label, plot_title):
        X, Y = np.meshgrid(self.lc_list, self.td_list)
        im = ax[i, j].imshow(data, cmap = plt.cm.jet,
                            vmin = data.min(), vmax = data.max(), origin = 'lower',
                            extent = [self.lc_list.min(), self.lc_list.max(),
                                    self.td_list.min(), self.td_list.max()])
        cbar_tick_delta = (data.max() - data.min())/10
        ax[i,j].scatter(X, Y, c=data, cmap =plt.cm.jet)
        divider = make_axes_locatable(ax[i,j])
        cax = divider.append_axes("right", size = "5%", pad = 0.1)
        plt.colorbar(im,cax = cax, ticks = MultipleLocator(cbar_tick_delta),
                     format = "%.2f")
        if set_x_label == True:
            ax[i,j].set_xlabel('cross_lenght')
        ax[i,j].set_ylabel('trig_day')
        ax[i,j].set_title(plot_title)

    def merge_data(self):
        self.merged_data = self.index_data[['volat_LC',
                                            'volat_slope',
                                            'volat_accel',
                                            'OSC_SLOW',
                                            'RSI']].join(self.max_min_fut_returns)
        self.merged_data = self.merged_data.join(self.rolling_lengths)


    def get_major_supp_resist(self):
        pass
      
    def correl_matrix(): ## computes current correl matrix between indeces returns and plots it
        pass




    def get_Kalman_avg(self, key, column_name):
        
        if key != 'close_EUR':
            
            TS = getattr( self.data_upd, key ).fillna(method = 'ffill').dropna()
            
        else:

            if ('close_EUR' in self.data_upd) & (self.data_upd.ticker[0] !='INDEXSP:.INX'):
                
                TS = self.data_upd.close_EUR.fillna(method = 'ffill').dropna()
                
            else:
                
                TS = self.data_upd.close.fillna(method = 'ffill').dropna()            
        
        kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = TS.values[0],
                      initial_state_covariance = 1,
                      observation_covariance = 1,
                      transition_covariance = .01)
        
        kalman_f = pd.DataFrame({'Kalman_filter':
                                 np.squeeze(
                                     kf.filter(TS.values)[0]
                                     )
                                 },
                                index = TS.index )
        self.data_upd['Kalman_filter_' + column_name ] = kalman_f



    def peak_recognition(self, TS_in, num_days, peak_trough, mean_type,plot):
        
        ''' Function finds all peaks or troughs of TS_in time series that meet specific conditions
        It also outputs pandas df with statistical info for TS/TS.mean() ratio
        
        Parameters:
        ==========================
        TS_in : pandas time series or dataframe
        num_days: integer. Num of days around edge to define a peak
        peak_trough : string to specify whether 'peak' or 'trough' are requested
        mean_type : string. 'fixed' or 'rolling'
        plot : boolean. True if a plot is requested
        
        Output:
        ===========================
        Pandas dataframe containing statistics for segment duration and associated returns
        '''
        
        if peak_trough == 'peak':
        ## PEAKS

            signal = (TS_in >= pd.rolling_max(TS_in,num_days)) & (TS_in >= pd.rolling_max(TS_in.shift(-num_days), num_days)) \
            & (TS_in > pd.rolling_mean(TS_in, 120) + pd.rolling_std(TS_in, 120))
            
            if mean_type == 'fixed':
                ratio = TS_in/(TS_in.mean() + TS_in.std())
            elif mean_type == 'rolling':
                ratio = TS_in/(pd.rolling_mean(TS_in,120) + pd.rolling_std(TS_in,120))
            if 'close_EUR' in self.data_upd.keys():
                segs_peak_rec = self.segs(ratio, self.data.close_EUR , 1.,'up')
            else:
                segs_peak_rec  = self.segs(ratio, self.data.close , 1.,'up')
                
        else:       
        ##TROUGHS
            signal = (TS_in <= pd.rolling_min(TS_in,num_days)) & (TS_in <= pd.rolling_min(TS_in.shift(-num_days),num_days)) \
            & (TS_in < pd.rolling_mean(TS_in,120) - pd.rolling_std(TS_in,120))
            
            if mean_type == 'fixed':
                ratio = TS_in/(TS_in.mean() - TS_in.std())
                
            elif mean_type == 'rolling':
                ratio = TS_in/(pd.rolling_mean(TS_in,120) + pd.rolling_std(TS_in,120))

            ratio= TS_in/(pd.rolling_mean(TS_in,120) - pd.rolling_std(TS_in,120))
            if 'close_EUR' in self.data_upd.keys():
                segs_peak_rec  = self.segs(ratio,self.data.close_EUR ,1.,'down')
            else:
                segs_peak_rec  = self.segs(ratio,self.data.close ,1.,'down')
            ##ratio_stats_down_STAND[ratio_stats_down_STAND.dur > 10].dur.plot(kind = 'bar')
                
        if plot == True:
            plt.plot(signal.nonzero()[0], TS_in[signal], 'ro')
            plt.plot(TS_in.values, label = 'TS')
            plt.plot(pd.rolling_mean(TS_in,120) + pd.rolling_std(TS_in,120), label ='roll_mean + one-sigma' )
            plt.plot(pd.rolling_mean(TS_in,120) - pd.rolling_std(TS_in,120), label = 'roll_mean - one-sigma')
            plt.legend()
        
        ##EXTRA  : ESTIMATE PEAK DAY FREQUENCY
#        f  = np.median(np.diff(TS_in[signal].index.values))
#        days = f.astype('timedelta64[D]').item().days
        
        return segs_peak_rec,TS_in[signal]





class Compute_stats(object):
    pass


class Email(object):
    pass


## how to profile a function line-by-line 
## %lprun -f fun_name fun_name(fun_args)


if __name__ == '__main__':
    ## make instance of data obj, then load and update data
    data_obj = Data_Input_Output() 
    data_obj.load_ALL_saved_data()
    data_obj.save_upd_data()
    ## make instance of compute obj
    ## load individual indices
    i = 9
    data_ind = Compute_indicators()
    data_ind.load_index(i)
    data_ind.comp_volatility(i)
    data_ind.comp_stoch_osc(i)
    data_ind.comp_RSI_osc(i)
    data_ind.comp_EMA_volat()
    data_ind.comp_EMA_index(i)
    data_ind.comp_volat_EMA_volat_ratio(i)

    ## define inputs
    indicator = data_ind.index_data.volat_EMA_volat_100.copy()
    if data_ind.FX_list[i] == 'local':
        price = data_ind.index_data.close_EUR.copy()
    else:
        price = data_ind.index_data.close.copy()
        
    volat = data_ind.index_data.volat_LC.copy()

    ## call segs method
    df_segs = data_ind.get_segs_data(indicator, price, 1.)

    ##plot
    df_segs.sort(['dur'],ascending=False).dur
    df_segs.sort(['mean_seg'])
    plt.hist(df_segs.sort(['dur'],ascending=False).dur.values)

    ## call track_volat method
    data_ind.track_volat_cycle(indicator, price, volat, 1.)

    plt.figure()
    ax1 = plt.subplot(511)
    ax1.plot(data_ind.index_data.index,data_ind.index_data.volat_LC,'b')
    ax2 = plt.subplot(512, sharex = ax1)
    ax2.plot(data_ind.index_data.index,data_ind.index_data.volat_EMA_volat_100,'r')
    ax3 = plt.subplot(513, sharex = ax1)
    ax3.plot(data_ind.index_data.index,data_ind.index_data.OSC_SLOW,'g')
    ax4 = plt.subplot(514, sharex = ax1)
    ax4.plot(data_ind.index_data.index,data_ind.index_data.RSI,'b')
    ax5 = plt.subplot(515, sharex = ax1)
    ax5.plot(data_ind.index_data.index,price,'b')

    ## call price_volat_pattern
    data_ind.get_price_volat_pattern(indicator, price, 1., 10, 'yes', 10, 'up')

    ## call plot price volat pattern
    data_ind.plot_volat_pattern(9, 2, 5)
    ## plot checking
    fig, ax =plt.subplots(3,1, sharex= True)
    ax[0].plot(df.indicator)
    ax[0].scatter(data_ind.index_data.index,data_ind.index_data.volat_EMA_volat_100, color='r')
    ax[1].plot(df.groups_trigg_stop_points)
    ax[2].plot(df.segs_roll_days)

    

    


































