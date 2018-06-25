import os
import numpy as np
from up_rise_example import Run_up_rise
from price_example import Run_price_1
from ETF_output import Gen_ETF_output
from etf_regression import output_etf_prediction_map

if __name__ == "__main__":
	#Generate Up & Rise
	up_rise_info = Run_up_rise()
	#Generate price (Model 1)
	price_1_info = Run_price_1()
	#Generate price (Model 2)
	price_2_info = output_etf_prediction_map()
	#Generate output CSV
	Gen_ETF_output(up_rise_info, price_1_info, price_2_info)