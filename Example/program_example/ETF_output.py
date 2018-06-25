import pandas as pd
import numpy as np
import os


# 未調整ETF tetfp
path = os.path.abspath(""+"../finance_data_example/tetfp.csv")

def Gen_ETF_dict(df, ETF_id_col, time_col):
    #Create a data frame dictionary to store each stock
    df[ETF_id_col] = df[ETF_id_col].str.strip()#.astype(str)
    ETFSymbol_list = df[ETF_id_col].unique().tolist()
    ETFDict = {elem : pd.DataFrame for elem in ETFSymbol_list}
    ETF_id = {elem.strip() : int for elem in ETFSymbol_list}
    for key in ETFDict.keys():
        ETFDict[key] =  df[df[ETF_id_col] == key].sort_values(by=time_col).drop([ETF_id_col], axis = 1)
        ETFDict[key][time_col] =  pd.to_datetime(ETFDict[key][time_col], format='%Y%m%d')
        ETFDict[key] = ETFDict[key].set_index(time_col)
        ETFDict[key].index.names = ['time']
        ETF_id[key] = ETFDict[key]['代碼'][0]

    return ETFDict, ETF_id


def Gen_ETF_output(up_rise_info, price_1_info, output_map):

    Tbrain_ETF = pd.read_csv(path, encoding="big5hkscs", thousands=',')
    (ETF_DICT, ETF_id) = Gen_ETF_dict(Tbrain_ETF, "中文簡稱", "日期")
    df = pd.read_csv('../finance_data_example/output_example.csv')
    columns_list = ['Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice', 'Wed_ud',
                    'Wed_cprice', 'Thu_ud', 'Thu_cprice', 'Fri_ud', 'Fri_cprice']
    price_list = ['Mon_cprice', 'Tue_cprice', 'Wed_cprice', 'Thu_cprice', 'Fri_cprice']
    ud_list = ['Mon_ud', 'Tue_ud', 'Wed_ud', 'Thu_ud', 'Fri_ud']
    ETFid_list = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]
    hi = []
    for x in np.nditer(price_1_info):
        hi.append(x)
    
    test = pd.DataFrame({'ETFid':ETFid_list})
    test = test.set_index('ETFid')

    #read ETF output map from FYC
    for i, val in enumerate(columns_list):
        test[val] = 0
    test['friday_close'] = 0
    for key in ETF_DICT.keys():
        test.loc[ETF_id[key], 'friday_close'] = ETF_DICT[key]['收盤價(元)'][-1]
        
    for idx in test.index:
        if idx in output_map.keys():
            for i, val in enumerate(price_list):
                test.loc[idx, val] = test.loc[idx, 'friday_close']  * (output_map[idx][0][i] + hi[i])/2
        else:
            for i, val in enumerate(price_list):
                test.loc[idx, val] = test.loc[idx, 'friday_close'] * hi[i]
        

    #adjust clost data
    for i, val in enumerate(price_list):
        for j, idx in enumerate(test.index):
            if(test.loc[idx,val] < 50):
                test.loc[idx,val] = round(test.loc[idx,val]*100)/100
            else:
                test.loc[idx,val] = round(test.loc[idx,val]*20)/20

    #read up_rise info
    up_rise_info = up_rise_info.astype(int)
    for i, val in enumerate(ud_list):
        test.loc[:, val] = up_rise_info[0,i]

    test = test.drop(columns=['friday_close'])

    #return output example
    test.to_csv('output_example.csv', index=True)


