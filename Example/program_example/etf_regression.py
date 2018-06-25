import talib
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
import datetime

def train_label_up_rise_5days(df):
    train_label = df.filter(['up_rise']).shift(-1)
    train_label = train_label.rename(columns={'up_rise': 'day1'})
    train_label['day2'] = train_label['day1'].shift(-1)
    train_label['day3'] = train_label['day2'].shift(-1)
    train_label['day4'] = train_label['day3'].shift(-1)
    train_label['day5'] = train_label['day4'].shift(-1)
    return train_label
def train_label_5days(df):
    train_label = df.filter(['close']).shift(-1)
    train_label = train_label.rename(columns={'close': 'day1'})
    train_label['day2'] = train_label['day1'].shift(-1)/df['close']
    train_label['day3'] = train_label['day1'].shift(-2)/df['close']
    train_label['day4'] = train_label['day1'].shift(-3)/df['close']
    train_label['day5'] = train_label['day1'].shift(-4)/df['close']
    train_label['day1'] = train_label['day1']/df['close']
    return train_label

def label_to_np(df):
    df = df.fillna(method='ffill')
    np_label = df.values
    #np_label[np_label >= 0] = 1
    #np_label[np_label < 0] = 0 
    return np_label

def concate_feature(feature_df, talib_out_np, talib_name, add_slope = False):
    
    feature_df[talib_name] = pd.DataFrame(talib_out_np, index= feature_df.index)
    if add_slope:
        temp = talib_out_np.copy()
        temp[1:,0] = temp[:-1,0]
        temp = talib_out_np - temp
        feature_df[talib_name+"_slope"] = pd.DataFrame(temp, index= feature_df.index)
    return feature_df

def rename_map(col_array, add_str):
    col_Dict = {str(elem) : "" for elem in col_array}
    for key in col_Dict.keys():
        col_Dict[key] = key + str(add_str)
    return col_Dict
    
#Run XGBoost and return test_label numpy (length, 5)
def auto_gen_Model(xgb_params, train_data, train_label, test_data, test_label):
    
    if type(train_data) != np.ndarray or type(train_label) != np.ndarray:
        print("Error: type of train_data_np and train_label_np need to be numpy.ndarray")
        return
    
    out_list = []
    label_list = []
    model_list = []
    for i in range(0,5):
        train = xgb.DMatrix(train_data, train_label[:,i])
        test_d = xgb.DMatrix(test_data, test_label[:,i])

        cv_output = xgb.cv(xgb_params, train, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=False, show_stdv=False
                         )
        model = xgb.train(dict(xgb_params, silent=1), train, num_boost_round= len(cv_output))
        
        out = model.predict(test_d)
        out = np.reshape(out, (out.shape[0], 1))
        out_list.append(out)
        model_list.append(model)
    
    #print (model.get_fscore())
    out_data = np.concatenate(out_list, axis = 1)
    return out_data, model_list

def Score(predict, label):
    if label.shape != predict.shape:
        return ("error: different shape ")
    
    score = np.abs(predict-label)        
    #cal = np.concatenate([cal, cal2], axis=1)
    mean = np.mean(score, axis=0)
    return mean
def No_model_score(label):
    predict = np.ones_like(label)
    score = np.abs(predict-label)        
    #cal = np.concatenate([cal, cal2], axis=1)
    mean = np.mean(score, axis=0)
    return mean
def output_etf_prediction_map():
    #US stock index
    US_path = os.path.abspath("../finance_data_example/")
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    name_list = ['NASDAQ', 'S&P500', 'DowJonesIndustrial', 'PHLX']
    df_list = []
    for i, val in enumerate(name_list):
        read_path = US_path + "/" + val + ".csv"
        df = pd.read_csv(read_path, parse_dates=['Date'], date_parser=dateparse, thousands=',').set_index('Date')
        name = val + '_up_rise'
        df[name] = (df['Adj Close'] - df['Adj Close'].shift(1))/df['Adj Close'].shift(1)
        df = df.filter([name])
        df_list.append(df)
    US_df = pd.concat(df_list, axis = 1)

    #3美股指數 up_rise 台股 3日 up_rise
    # KD? RSI 5 RSI 10?
    TWSI_path = os.path.abspath("../finance_data_example/TaiwanExchange3.csv")
    TWSI = pd.read_csv(TWSI_path, encoding='big5', parse_dates=['time'], date_parser=dateparse, thousands=',').set_index('time')
    talib_inputs = {
    'open': TWSI['open'].as_matrix().astype(float),
    'high': TWSI['high'].as_matrix().astype(float),
    'low': TWSI['low'].as_matrix().astype(float),
    'close': TWSI['close'].as_matrix().astype(float),
    'volume': TWSI['成交金額'].as_matrix().astype(float),
    #'外資': TWSI['外資買賣差額'].as_matrix().astype(float)
    }
    TWSI = TWSI.filter(['close', '漲跌點數'])
    TWSI['up_rise'] = (TWSI['close'] - TWSI['close'].shift(1))/TWSI['close'].shift(1)
    TWSI['up_rise_4'] = (TWSI['close'] - TWSI['close'].shift(4))/TWSI['close'].shift(4)



    #ffill nan 漲跌
    US_df_adjust = pd.concat([TWSI, US_df], axis=1)#*100
    US_df_adjust = US_df_adjust.filter([val + '_up_rise' for i, val in enumerate(name_list)])
    US_df_adjust = US_df_adjust.fillna(method='ffill')

    feature = pd.concat([TWSI, US_df_adjust], axis=1, join_axes=[TWSI.index])

    #KD
    slowk, slowd = talib.abstract.STOCH(talib_inputs,5,3,0,3,0)
    slowk = np.reshape(slowk, (slowk.shape[0], 1))
    slowd = np.reshape(slowd, (slowd.shape[0], 1))
    feature = concate_feature(feature, slowk, 'slowk', True)
    feature = concate_feature(feature, slowd, 'slowd', False)
    feature['k/d'] = feature['slowk'] / feature['slowd'] #> 1

    #MA
    ma = talib.MA(talib_inputs['close'], 4)
    ma = np.reshape(ma, (ma.shape[0], 1))
    #feature = concate_feature(feature, ma, 'ma', True)
    #feature = feature.drop(columns='ma')

    #MA 60
    ma60 = talib.MA(talib_inputs['close'], 60)
    ma60 = np.reshape(ma60, (ma60.shape[0], 1))
    feature = concate_feature(feature, ma60, 'ma60', True)
    feature['多頭'] = feature['close'] / feature['ma60'] 
    feature['多頭'] = feature['多頭'] > 1
    #BBAND
    up, mid, low = talib.BBANDS(talib_inputs['close'])
    up = up/talib_inputs['close']
    up = up < 1
    low = low/talib_inputs['close']
    low = low > 1
    mid = mid/talib_inputs['close']
    up = np.reshape(up, (up.shape[0], 1))
    low = np.reshape(low, (low.shape[0], 1))
    mid = np.reshape(mid, (up.shape[0], 1))
    feature = concate_feature(feature, up, 'up', False)
    feature = concate_feature(feature, low, 'low', False)
    feature = concate_feature(feature, mid, 'mid', False)


    feature = feature.drop(columns=['close', '漲跌點數', 'slowd', 'ma60'])
    feature.head(n=3)

    #Only keep friday and Saturday?
    F_or_S_idx = np.array(list(filter(lambda x: x.weekday() == 4 or x.weekday() == 5, feature.index)))
    step = datetime.timedelta(days=1)
    redundant_F_idx = []
    for x in np.ndenumerate(F_or_S_idx):
        if(F_or_S_idx[x[0]].weekday() == 5 and F_or_S_idx[x[0]] - step == F_or_S_idx[x[0][0] - 1]):
            redundant_F_idx.append(x[0][0] - 1)
    F_or_S_idx = np.delete(F_or_S_idx, redundant_F_idx)

    #Gen label np # Note: 最後一個 label 可能是未來NAN
    df_train_label = train_label_5days(TWSI)
    df_train_label_only_friday = pd.DataFrame(df_train_label, index = F_or_S_idx)
    np_exchange_label = label_to_np(df_train_label)
    #Gen feature np
    feature_only_friday = pd.DataFrame(feature, index = F_or_S_idx)
    np_feature = feature.values
    np_feature_only_friday = feature_only_friday.values

    ETF_path = os.path.abspath("../finance_data_example/taetfp.csv")
    dateparse2 = lambda x: pd.datetime.strptime(x, '%Y%m%d')
    ETF = pd.read_csv(ETF_path, encoding='big5', parse_dates=['日期'], date_parser=dateparse2, thousands=',').set_index('日期')
    ETF['close'] = ETF['收盤價(元)']
    #ETF.loc[ETF['代碼'] == 50]
    count = ETF.groupby('代碼').代碼.count()
    count = count[count < 1000]
    #ETF
    for code in count.index:
        ETF = ETF[ETF.代碼 != code]
        ETF = ETF.filter(['代碼','收盤價(元)','close'])
    ETF_list = []
    count = ETF.groupby('代碼').代碼.count()
    for code in count.index:
        ETF_list.append(ETF[ETF['代碼']==code])
        ETF_list[-1]['up_rise'] = (ETF_list[-1]['收盤價(元)'] - ETF_list[-1]['收盤價(元)'].shift(1))/ETF_list[-1]['收盤價(元)'].shift(1)
        ETF_list[-1]['up_rise_4'] = (ETF_list[-1]['收盤價(元)'] - ETF_list[-1]['收盤價(元)'].shift(4))/ETF_list[-1]['收盤價(元)'].shift(4)

    df_label_list = []
    df_feature_list = []
    df_label_list_only_friday = []
    df_feature_list_only_friday = []
    for i in range(len(ETF_list)):
        ETF_df_train_label = train_label_5days(ETF_list[i])
        ETF_F_or_S_idx = np.array(list(filter(lambda x: x.weekday() == 4 or x.weekday() == 5, ETF_list[i].index)))
        step = datetime.timedelta(days=1)
        redundant_F_idx = []
        for x in np.ndenumerate(ETF_F_or_S_idx):
            if(ETF_F_or_S_idx[x[0]].weekday() == 5 and ETF_F_or_S_idx[x[0]] - step == ETF_F_or_S_idx[x[0][0] - 1]):
                redundant_F_idx.append(x[0][0] - 1)
        ETF_F_or_S_idx = np.delete(ETF_F_or_S_idx, redundant_F_idx)
        ETF_df_train_label_only_friday = pd.DataFrame(ETF_df_train_label, index = ETF_F_or_S_idx)
        feature_only_friday = pd.DataFrame(ETF_list[i], index = ETF_F_or_S_idx)
        feature_only_friday = feature_only_friday.filter(['up_rise','up_rise_4'])
        feature = ETF_list[i].filter(['up_rise','up_rise_4'])
        feature = feature.values
        feature_only_friday = feature_only_friday.values
        df_feature_list.append(feature)
        df_feature_list_only_friday.append(feature_only_friday)
        np_train_label = label_to_np(ETF_df_train_label)
        np_train_label_only_friday = label_to_np(ETF_df_train_label_only_friday)
        df_label_list.append(np_train_label)
        df_label_list_only_friday.append(np_train_label_only_friday)

    #Training
    xgb_params = {
    'eta': 0.15,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
    }
    remove_idx = 0#450#645
    idx = -10 #for training
    np_feature = np_feature[-1337:]
    sums = 0
    ref_sum = 0
    last_num = -2
    ETF_output_map = {}
    for i,l in enumerate(df_label_list):
        #print(len(df_feature_list[i]))
        #print(len(np_feature))
        np_feature3 = np.concatenate((np_feature,df_feature_list[i]),axis = 1)
        train = np_feature3[:idx]
        test = np_feature3[idx:]
        np_label2 = l[remove_idx:]
        train_label = np_label2[:idx]
        test_label = df_label_list_only_friday[i][last_num:] #-2
        np_feature3_only_friday = np.concatenate((np_feature_only_friday[-260:],df_feature_list_only_friday[i]),axis = 1)
        test = np_feature3_only_friday[last_num:]
        out_data, model_list = auto_gen_Model(xgb_params, train, train_label, test, test_label)
        mean = Score(out_data, test_label)
        #ref_mean = No_model_score(test_label)
        #print("average[",i,"]:", mean.sum()/5)
        #print("ref average[",i,"]:", ref_mean.sum()/5)
        ETF_output_map[ETF_list[i]['代碼'][0]] = out_data[-1:]
        print(ETF_list[i]['代碼'][0],out_data[-1:])
        #sums+=mean/5
        #ref_sum+=ref_mean/5
    return ETF_output_map
if __name__ == "__main__":
    output_etf_prediction_map()
