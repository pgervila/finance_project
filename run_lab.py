## import and reload index module
import finance_lab as idx_model
from imp import reload
reload(idx_model)

# reset seaborn
# seaborn.reset_orig()
# %pylab

##import libraries
import pandas as pd
import matplotlib.cm as cm

## update and save data
data_obj = idx_model.Data_Input_Output() 
data_obj.load_ALL_saved_data()
data_obj.save_upd_data()

## make instance of compute obj
data_ind = idx_model.Compute_indicators()
## load individual index and compute indicators
i = 9
data_ind.load_index(i)
data_ind.comp_volatility(i)
data_ind.comp_stoch_osc(i)
data_ind.comp_RSI_osc(i)
data_ind.comp_EMA_volat()
data_ind.get_volat_slope(5,20)
data_ind.comp_EMA_index(i)
data_ind.comp_volat_EMA_volat_ratio(i)
## define inputs
indicator = data_ind.index_data.volat_EMA_volat_100.copy()
indicator_volat_slope = data_ind.index_data.volat_slope.copy()
if data_ind.FX_list[i] == 'local':
    price = data_ind.index_data.close_EUR.copy()
else:
    price = data_ind.index_data.close.copy()    
volat = data_ind.index_data.volat_LC.copy()
##extra_data and MERGE
ret_period = 100
data_ind.get_max_min_fut_return_in_period(i, ret_period)
data_ind.find_roll_present_past_segs_length(indicator)
data_ind.merge_data()

##################################################

## TODO: add params to filter potentially interesting points
def pattern_finder(data_ind):
    data = data_ind.merged_data[['volat_LC', 'max_ret_in_period', 'min_ret_in_period',
                                 'volat_slope', 'volat_accel', 'RSI',
                                 'OSC_SLOW', 'counts', 'sign',
                                 'lag_2', 'lag_4']].copy()
    data = data[ (data.volat_slope <= 0) & (data.RSI < 50) &
                 (data.volat_LC > pd.rolling_mean(data.volat_LC, 100))]
    data = data.dropna()
    ret_data = data['max_ret_in_period'].values
    to_plot = [['RSI','OSC_SLOW','volat_LC'],
               ['lag_2','lag_4','volat_slope']]

    return data, ret_data, to_plot

data, ret_data, to_plot = pattern_finder(data_ind)
cm = plt.cm.get_cmap('RdYlBu')

fig, axes = plt.subplots(2, 3, sharex = True)
for idx_1, elem_1 in enumerate(to_plot):
    for idx_2, elem_2 in enumerate(elem_1):    
        cbar_tick_delta = (ret_data.max() - ret_data.min())/10
        sc = axes[idx_1, idx_2].scatter(data['counts'],
                                        data[elem_2],
                                        c = ret_data,
                                        s = 30)
        axes[idx_1, idx_2].set_ylabel(elem_2)
        axes[idx_1, idx_2].yaxis.labelpad = 0
        axes[idx_1, idx_2].tick_params(direction = 'in',
                                       labelsize = 'small')
        axes[idx_1, idx_2].grid()
        
fig.text(0.4, 0.04, 'roll_days_in_pos_volat_seg (COMMON AXIS)')
plt.suptitle('max_ret_in_period = ' + str(ret_period) + ' days',
             fontsize = 16)
fig.subplots_adjust(left = 0.05, right = 0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.7])
fig.colorbar(sc, cax = cbar_ax)


##plot index overview
fig, axes = plt.subplots(2,2, sharex = True)
axes[0, 0].plot(data_ind.index_data.close)
axes[0, 0].set_ylabel('close')
axes[0, 0].grid()
axes[0, 1].plot(data_ind.index_data.EMA_volat_20, label = 'EMA_20')
axes[0, 1].plot(data_ind.index_data.EMA_volat_50, label = 'EMA_50')
axes[0, 1].set_ylabel('EMA_volat')
axes[0, 1].legend()
axes[0, 1].grid()
axes[1, 0].plot(data_ind.index_data.volat_slope,
                marker='o',
                markersize = 2)
axes[1, 0].axhline(0)
axes[1, 0].set_ylabel('volat_slope')
axes[1, 0].grid()
axes[1, 1].plot(data_ind.index_data.RSI, color = 'g')
axes[1, 1].axhline(70, color = 'k')
axes[1, 1].axhline(30, color = 'k')
axes[1, 1].set_ylabel('RSI')
axes[1, 1].grid()
fig.text(0.5, 0.04, 'date')



## volat_slope segments, plotting
segs_volat_slope = data_ind.get_segs_data(indicator_volat_slope, price, threshold = -0.1)
volat_slope_lags = data_ind.find_roll_present_past_segs_length(indicator_volat_slope,
                                                               threshold = -0.1)
ret_period = 100
max_min_rets = data_ind.get_max_min_fut_return_in_period(12, ret_period)
rets_and_lags = max_min_rets.join(volat_slope_lags)
rets_and_lags = rets_and_lags.join(data_ind.index_data[['EMA_volat_10',
                                                        'EMA_volat_20',
                                                        'OSC_SLOW',
                                                        'RSI']])
df = rets_and_lags[['max_ret_in_period','lag_1','lag_2','OSC_SLOW','RSI']]
plt.hexbin(df.lag_1,df.lag_2, C = df.max_ret_in_period)
plt.colorbar()

##############################################################"
## volat-slope filtered picked points
## visualize HIGH POTENTIAL RETURN POINTS

## update volat slope and accel
data_ind.get_volat_slope(10, 10)
##extra_data and MERGE
data_ind.merged_data['volat_slope'] = data_ind.index_data.volat_slope
data_ind.merged_data['volat_accel'] = data_ind.index_data.volat_accel
##define new indicator based on volat slope
indicator_volat_slope = data_ind.index_data.volat_slope.copy()

seg_matrix = data_ind.get_segs_data(indicator_volat_slope,
                                    price,
                                    threshold = -0.1)
seg_matrix = seg_matrix[seg_matrix.dur < 0]
## filter starting points from data
high_pot_ret_points = data_ind.merged_data.ix[seg_matrix.start].copy()
high_pot_ret_points = high_pot_ret_points[ (high_pot_ret_points.RSI < 70)]


## visualize HIGH POTENTIAL RETURN POINTS 
fig, ax = plt.subplots(2, 3, sharex = True)
point_size = 10
## plot 1
ax[0, 0].plot(data_ind.index_data.close, label = 'index_price')
ax[0, 0].scatter(high_pot_ret_points.index,
            high_pot_ret_points.prices,
            color = 'r', s = 10, label = "high_pot_ret_points")
ax[0, 0].grid()
ax[0, 0].set_ylabel('price')
ax[0, 0].legend(loc = 'best', fontsize = 'x-small')
## plot 2
ax[0, 1].plot(data_ind.index_data.RSI, 'k--',
              c = 'g', color = 'r', label ='RSI')
ax[0, 1].scatter(high_pot_ret_points.index, high_pot_ret_points.RSI,
                 s = point_size, color = 'r',
                 label = "high_pot_ret_points")
ax[0, 1].axhline(70,
                 ls = 'dashed', c = 'red')
ax[0, 1].axhline(30,
                 ls = 'dashed', c = 'blue')
ax[0, 1].grid()
ax[0, 1].set_ylabel('RSI')
ax[0, 1].legend(loc = 'best', fontsize = 'x-small')
## plot 3
ax[0, 2].plot(data_ind.index_data.OSC_SLOW, 'k--', c = 'g')
ax[0, 2].scatter(high_pot_ret_points.index,
                 high_pot_ret_points.OSC_SLOW,
                 s = point_size, color = 'r',
                 label = "high_pot_ret_points")
ax[0, 2].axhline(80,
                 ls = 'dashed', c = 'red')
ax[0, 2].axhline(20,
                 ls = 'dashed', c = 'blue')
ax[0, 2].grid()
ax[0, 2].set_ylabel('OSC_SLOW')
ax[0, 2].legend(loc = 'best', fontsize = 'x-small')
## plot 4
ax[1, 0].scatter(high_pot_ret_points.index, high_pot_ret_points.volat_LC,
                 s = point_size, color = 'r',
                 label = "high_pot_ret_points" )
ax[1, 0].axhline(data_ind.index_data.volat_LC.mean(),
                 ls = 'dashed', c = 'blue',
                 label = "volat_mean")
ax[1, 0].axhline(data_ind.index_data.volat_LC.min(),
                 ls = 'dashed', c = 'red',
                 label = "volat_min")
ax[1, 0].plot(data_ind.index_data.volat_LC, label = 'volat')
ax[1, 0].plot(data_ind.index_data.EMA_volat_100, 'k--',
              color = 'g', label = 'EMA_volat_100')
ax[1, 0].set_ylabel('volat_LC')
ax[1, 0].grid()
ax[1, 0].legend(loc = 'best', fontsize = 'x-small')
## plot 5
ax[1, 1].scatter(high_pot_ret_points.index, high_pot_ret_points.volat_EMA_volat_100,
                 s = point_size, color = 'r',
                 label = "high_pot_ret_points" )
ax[1, 1].plot(data_ind.index_data.volat_EMA_volat_100, color = 'blue',
              label = "volat_EMA_volat_100")
ax[1, 1].axhline(1, ls = 'dashed', c = 'blue')
ax[1, 1].axhline(data_ind.index_data.volat_EMA_volat_100.min(),
                 ls = 'dashed', c = 'red', label = 'min')
ax[1, 1].set_ylabel('volat_EMA_volat_100')
ax[1, 1].grid()
ax[1, 1].legend(loc = 'best', fontsize = 'x-small')
## plot 6
ax[1, 2].bar(high_pot_ret_points.index, high_pot_ret_points.counts,
             width = 1, color = 'r',
             label = "high_pot_ret_points")
ax[1, 2].plot(data_ind.merged_data.counts, '--k',
              color = 'blue')
ax[1, 2].set_ylabel('roll_days_seg')
ax[1, 2].grid()
ax[1, 2].legend(loc = 'best', fontsize = 'x-small')

## adjust subplots position
fig.subplots_adjust(left = 0.05, right = 0.95)

#######################################################################"

### OPTIMIZATION
import scipy.optimize as sp_opt

# define funct that computes returns in segs as function of
# days to compute volat_slope and EMA length to compute EMA_volat
def fun_to_maximize(n, EMA_days):
    """ returns seg ret and seg_max_ret
        as function of :
        n : # days used to compute volat_slope
        EMA_days : EMA length in days used to compute EMA_volat"""
    data_ind.get_volat_slope(n, EMA_days)
    indicator_volat_slope = data_ind.index_data.volat_slope.copy()
    # call get_segs_data method from Compute_indicators class
    seg_matrix = data_ind.get_segs_data(indicator_volat_slope,
                                        price,
                                        threshold = -0.1)
    # keep only segs for which indicator is below threshold
    seg_matrix = seg_matrix[seg_matrix.dur < 0]
    # return cum_ret and max_cum_ret for selected segments
    return seg_matrix.cum_ret.mean(), seg_matrix.max_cum_ret.mean()

def compute_idx_data( i, n, EMA_days):
    data_ind.load_index(i)
    data_ind.comp_volatility(i)
    data_ind.comp_stoch_osc(i)
    data_ind.comp_RSI_osc(i)
    data_ind.comp_EMA_volat()
    data_ind.get_volat_slope(n, EMA_days)
    data_ind.comp_EMA_index(i)
    data_ind.comp_volat_EMA_volat_ratio(i)

    ## define inputs
    indicator = data_ind.index_data.volat_EMA_volat_100.copy()
    indicator_volat_slope = data_ind.index_data.volat_slope.copy()
    if data_ind.FX_list[i] == 'local':
        price = data_ind.index_data.close_EUR.copy()
    else:
        price = data_ind.index_data.close.copy()      
    volat = data_ind.index_data.volat_LC.copy()

    ##extra_data and MERGE
    ret_period = 100
    data_ind.get_max_min_fut_return_in_period(i, ret_period)
    data_ind.find_roll_present_past_segs_length(indicator)
    data_ind.merge_data()
    return indicator, indicator_volat_slope, price, volat

indicator, indicator_volat_slope, price, volat = compute_idx_data( 12, 20, 50)

# build 2D arrays to visualize fun_to_maximize results
n_list = np.arange(1,41, 2)
EMA_days_list = np.array([10, 20, 50])
ret_list = np.zeros((len(n_list), len(EMA_days_list)))
max_ret_list = np.zeros((len(n_list), len(EMA_days_list)))
for ix_1, n in enumerate(n_list):
    for ix_2, EMA in enumerate(EMA_days_list):
        fun_output = fun_to_maximize(n, EMA)
        ret_list[ix_1, ix_2] = 100*fun_output[0]
        max_ret_list[ix_1, ix_2] = 100*fun_output[1]

var_to_plot = ret_list

plt.plot(n_list,var_to_plot[:,0], marker = 'o', label ='EMA_10')
plt.plot(n_list,var_to_plot[:,1], marker = 'o', label ='EMA_20')
plt.plot(n_list,var_to_plot[:,2], marker = 'o', label ='EMA_50')
plt.xlabel('n_days_delta_volat_slope')
plt.ylabel('segs cum_ret')
plt.legend('best')

# CONCLUSION : using more days to compute volat_slope leads to better expectations
# in max_ret



#  2D PLOT
X, Y = np.meshgrid(EMA_days_list, n_list)
plt.imshow(max_ret_list, cmap = plt.cm.jet, extent = [X.min(),X.max(),Y.max(),Y.min()])
plt.xlabel('EMA_length')
plt.ylabel('n_days_diff_volat_slope')
plt.colorbar()

# compute rets and max_rets for optimum input combination
data_ind.get_volat_slope(20,50)
indicator_volat_slope = data_ind.index_data.volat_slope.copy()
segs_data_all = data_ind.get_segs_data(indicator_volat_slope, price, threshold = -0.1)
segs_data = segs_data_all[ segs_data_all.dur < 0 ]
segs_data.cum_ret.plot(marker='o')
segs_data.max_cum_ret.plot(marker='o')
segs_data.max_cum_ret.mean()
pct_useless_segs = len(np.where(segs_data.max_cum_ret == 0.0)[0])/len(segs_data)

# plot seg dependency
plt.scatter(segs_data_all.dur,segs_data_all.dur.shift())

# plot key points
price.plot()
plt.scatter(bb.start.values,price[bb.start.values],s=50,c='r')
# plot scatter, find best
data_ind.get_volat_slope(2,10)
vol_vel_2_10 = data_ind.index_data.volat_slope.copy()
vol_acc_2_10 = data_ind.index_data.volat_accel.copy()
data_ind.get_volat_slope(2,20)
vol_vel_2_20 = data_ind.index_data.volat_slope.copy()
vol_acc_2_20 = data_ind.index_data.volat_accel.copy()
data_ind.get_volat_slope(20,50)
vol_vel_20_50 = data_ind.index_data.volat_slope.copy()
vol_acc_20_50 = data_ind.index_data.volat_accel.copy()
plt.scatter(vol_vel_2_10[bb.start.values], vol_vel_2_20[bb.start.values],
            c=100*bb.max_cum_ret, s = 50)
plt.colorbar()

plt.scatter(vs_2_10[bb.start.values], vol_acc_2_20[bb.start.values],
            c=100*bb.max_cum_ret, s = 50)

plt.scatter(vol_vel_20_50[bb.start.values],
            vol_acc_20_50[bb.start.values],
            c = 100*bb.max_cum_ret, s = 50,
            cmap = cmap_discretize( 'jet', 20))


plt.scatter(data_ind.merged_data['counts'][bb.start.values],
            vol_vel_2_20[bb.start.values],
            c=100*bb.max_cum_ret, s = 50)

# filter out segs whose length is less than five
cc= bb[bb.dur<-5]
plt.scatter(data_ind.merged_data['counts'][bb.start.values],
            data_ind.index_data.EMA_volat_20[bb.start.values],
            c = 100*bb.max_cum_ret, s = 50,
            cmap = cmap_discretize( 'jet', 20))

# DISCRETIZE CMAP
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """    
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = concatenate((linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


## PLOT ALL IDX
fig, ax = plt.subplots(4,4)


xx,yy = np.meshgrid(np.arange(4),np.arange(4))
xx.transpose()
xx_t = xx.transpose().reshape(16)
yy_t = yy.transpose().reshape(16)
ax_list = list(zip(xx_t,yy_t))


for i,ax_loc in enumerate(ax_list[:-2]):

    indicator, indicator_volat_slope, price, volat = compute_idx_data( i, 20, 50)

    # build 2D arrays to visualize fun_to_maximize results
    n_list = np.arange(1,21,2)
    EMA_days_list = np.array([10,20,50])
    ret_list = np.zeros((len(n_list), len(EMA_days_list)))
    max_ret_list = np.zeros((len(n_list), len(EMA_days_list)))
    for ix_1, n in enumerate(n_list):
        for ix_2, EMA in enumerate(EMA_days_list):
            fun_output = fun_to_maximize(n, EMA)
            ret_list[ix_1, ix_2] = 100*fun_output[0]
            max_ret_list[ix_1, ix_2] = 100*fun_output[1]

    var_to_plot = max_ret_list

    ax[ax_loc].plot(n_list,var_to_plot[:,0], marker = 'o', label ='EMA_10')
    ax[ax_loc].plot(n_list,var_to_plot[:,1], marker = 'o', label ='EMA_20')
    ax[ax_loc].plot(n_list,var_to_plot[:,2], marker = 'o', label ='EMA_50')
    ax[ax_loc].set_xlabel('n_days_delta_volat_slope')
    ax[ax_loc].set_ylabel('segs_max_cum_ret')
    ax[ax_loc].legend(loc='best', fontsize = 'small')
    print(i)


## ADD SUPPORT / RESISTANCE INFLUENCE / ANALYSIS

def get_supp_resist(input_TS, num_days = 50, pct_gap = 0.05, peak = True):
    ## VI !!!!!!!!!!!!!!!!!!!!!!!!
    ##TODO : assign quant label to points based on vert % distance btw point/max(val(num_days)-point)
    if type(input_TS) != pd.core.series.Series:
        input_TS = pd.Series(input_TS)

    if peak == True:
        ## find resistance points
        signal = ((input_TS >= pd.rolling_max(input_TS, num_days)) &
                  (input_TS >= pd.rolling_max(input_TS.shift(-num_days), num_days)) &
                  (input_TS >= (1 + pct_gap) * pd.rolling_min(input_TS, 2 * num_days)) &
                  (input_TS >= (1 + pct_gap) * pd.rolling_min(input_TS.shift(-2 * num_days), 2 * num_days)))

        signal_strengths = []
        for idx, supp_point in input_TS[signal].iteritems():
            iloc = input_TS.index.get_loc(idx)
            strength_1 = supp_point/input_TS[iloc - 100 : iloc ].min()
            strength_2 = supp_point/input_TS[iloc : iloc + 100].min()
            strength = (strength_1 + strength_2) / 2
            signal_strengths.append(strength)
        signal_strengths = np.array(signal_strengths)       
    else:
        ## find support points
        signal = ((input_TS <= pd.rolling_min(input_TS, num_days)) &
                  (input_TS <= pd.rolling_min(input_TS.shift(-num_days), num_days)) &
                  (input_TS <= (1 - pct_gap) * pd.rolling_max(input_TS, 2 * num_days)) &
                  (input_TS <= (1 - pct_gap) * pd.rolling_max(input_TS.shift(- 2 * num_days), 2 * num_days)))               
        signal_strengths = []
        for idx, supp_point in input_TS[signal].iteritems():
            iloc = input_TS.index.get_loc(idx)
            strength_1 = 1/(supp_point/input_TS[iloc - 100 : iloc ].max())
            strength_2 = 1/(supp_point/input_TS[iloc : iloc + 100].max())
            strength = (strength_1 + strength_2) / 2
            signal_strengths.append(strength)
        signal_strengths = np.array(signal_strengths)
    return signal, signal_strengths

def get_correct_price(i):
    if data_ind.FX_list[i] == 'local':
        price = data_ind.index_data.close_EUR.copy()
    else:
        price = data_ind.index_data.close.copy()
    return price



input_TS = get_correct_price(i)
my_peaks, my_sign_strength = get_supp_resist(input_TS, num_days = 25, peak = True)
plt.figure()
plt.plot(input_TS)
plt.scatter(input_TS[my_peaks].index, input_TS[my_peaks],
            s = 40, c = my_sign_strength,
            cmap = plt.cm.get_cmap('viridis'))
plt.colorbar()
plt.grid()

pdt = input_TS[my_peaks].index.to_pydatetime()
sdt = np.vectorize(lambda s: s.strftime('%Y%m%d'))(pdt)
idt = sdt.astype('int')/100000
X = np.array(list(zip(idt, input_TS[my_peaks].values)))
plt.figure()
plt.scatter(X[:,0],X[:,1], s =10)



## K-MEANS
from sklearn.cluster import KMeans

n_clusters = 5

km = KMeans(n_clusters = n_clusters, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.figure()
markers = ['s','o','v','p','d']
colors = ['lightgreen','orange','lightblue','blue','black']

for idx, clust_label in enumerate(np.unique(y_km)):
    plt.scatter(X[y_km == clust_label, 0], 
                X[y_km == clust_label, 1], 
                s = 50, 
                c = colors[idx], 
                marker = markers[idx], 
                label='clus_' + str(idx))
plt.scatter(km.cluster_centers_[:,0], 
            km.cluster_centers_[:,1], 
            s=250, 
            marker='*', 
            c='red', 
            label='centrs')
plt.legend(loc = 'best')
plt.grid()

##IDENTIFY POINTS THAT ARE BOTH SUPPPORT AND RESISTANCES => strong points
