import pandas as pd
import numpy as np


############
#Given a list to be binned(input_list) and a list of bin values(bin_list, 
#this function will the return where each element in the input_list falls 
#belong to. If input_list is extremely long, writing a for loop or using 
#pd.Series.apply may not be fast enough. This function make use of the 
#pandas vectorized operations to make this binning fast.
#Parameters:
#input_list: a python list to be binned
#bin_list: a python list wit all the bin values
#logical_min: a float for extreme lower bound
#logical_max: a float for extreme upper bound
#Output:
#res_df: a dataframe with 3 columns('input','lowerbound','upperbound')
def binning_func(input_list,bin_list,logical_min=-np.Inf,logical_max=np.Inf):
    values = pd.Series(input_list)
    val_df = pd.DataFrame({'val':values})

    bin_values = pd.Series(list(set(bin_list)))
    bin_values.sort_values(inplace=True)
    bin_val_df = pd.DataFrame({'bin_val':bin_values})
    bin_val_df['upperbound'] = bin_val_df.bin_val.shift(-1).fillna(logical_max)
    bin_val_df.set_index('bin_val',inplace=True)

    new_cols = []
    for bv in bin_values:
      val_df[f'{bv}'] = bv < val_df['val']
      new_cols.append(f'{bv}')

    mask_df = np.exp(pd.DataFrame([bin_list]*len(input_list),columns=new_cols))
    masked_df = val_df[new_cols] * mask_df[new_cols]
    lowerbounds = np.log(masked_df.max(axis=1)).replace(-np.Inf,logical_min)

    res_df = pd.DataFrame({
        'input':input_list,
        'lowerbound':lowerbounds,
    }).join(bin_val_df,on='lowerbound')

    res_df['upperbound'] = res_df['upperbound'].fillna(bin_values.min())

    return res_df


if __name__ == '__main__':
    ######sample usage 1:
    ######output:
    ######    input   lowerbound  upperbound
    ######    -1.2    -50.0       -1.1
    ######    -0.3    -1.1        0.0
    ######    1.3      0.0        1.3
    ######    3.0      2.0        3.0
    ######    3.1      3.0        inf
    input_list = [-1.2,-0.3,1.3,3.0,3.1]
    bin_list = [-1.1,0.,1.3,1.5,2.,3.]
    logical_min = -50.
    logical_max = np.Inf

    binning_func(input_list,bin_list,logical_min,logical_max)
