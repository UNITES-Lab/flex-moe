import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

# Required Files
bio_1 = pd.read_csv('./APOERES_09May2024.csv')
bio_2 = pd.read_csv('./UPENNBIOMK_ROCHE_ELECSYS_09May2024.csv')
label = pd.read_csv('../diagnosis/DXSUM_PDXCONV_22Apr2024.csv')

def process_num_cat_cols(_df, initial_filling=None, col_dict=None):
    df_og = _df.copy()
    df = _df.copy()

    categorical_col_idx = []
    numerical_col_idx = []
    date_col_idx = []
    pattern = r'[-:\sa-zA-Z]'

    for col in df.columns:
        if (df[col].dtypes == int) or (df[col].dtypes == float):
            # If unique values are two, consider it categorical
            if df[col].nunique() <= 2:
                categorical_col_idx.append(col)
            else:
                numerical_col_idx.append(col)
        elif (pd.api.types.is_datetime64_any_dtype(df[col])) or ('DATE' in col.upper()):
            date_col_idx.append(col)
        else:
            if df[col].astype(str).str.contains(pattern).any():
                categorical_col_idx.append(col)
            else:
                numerical_col_idx.append(col)
    
    cat_missing_cols = []
    df_og = pd.concat([df[numerical_col_idx], df[categorical_col_idx], df[date_col_idx]], axis=1)
    if col_dict:
        df_og.columns = [col_dict[k] for k in list(df_og.columns)]

    ## For numerical columns
    if len(numerical_col_idx) > 0:
        print(f'Processing Numerical columns... {len(numerical_col_idx)}/{df_og.shape[1]}')
        df_num = df[numerical_col_idx][:]
        for column in df_num.columns:
            df_num[column] = df_num[column].astype('str').str.replace('<', '').str.replace('>', '').astype(float) 

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_scaled = min_max_scaler.fit_transform(df_num)
        df_num = pd.DataFrame(x_scaled, columns=numerical_col_idx, index=df.index)
        
        if df_num.isnull().sum().sum() > 0:
            if initial_filling == 'mean':
                df_num = df_num.fillna(df_num.mean().iloc[0])
            elif initial_filling == 'median':
                df_num = df_num.fillna(df_num.median().iloc[0])

    ## For categorical columns
    if len(categorical_col_idx) > 0:
        print(f'Processing Categorical columns... {len(categorical_col_idx)}/{df_og.shape[1]}')
        df_cat_ = df[categorical_col_idx][:]
        unique_value_counts = df_cat_.nunique()
        cut_off_criterion = unique_value_counts < 1000
        df_cat_ = df_cat_.loc[:, cut_off_criterion]  # filter columns to prevent dimension exploding
        categorical_col_idx = list(df_cat_.columns)

        if df_cat_.isnull().sum().sum() > 0:
            cat_missing_cols = list(df_cat_.columns[df_cat_.isnull().sum() > 0])
            if initial_filling != 'None':
                df_cat_.fillna(df_cat_.mode().iloc[0], inplace=True)
            
        df_cat = pd.get_dummies(df_cat_, columns=categorical_col_idx)
        df_cat = df_cat.replace({True: 1, False: -1})

        if initial_filling == 'None':
            if len(cat_missing_cols) > 0:
                for j in cat_missing_cols:
                    idx = np.where(df_cat.loc[:, df_cat.columns.str.startswith(f'{j}_')].sum(1) == 0)
                    df_cat.iloc[idx[0], df_cat.columns.str.startswith(f'{j}_')] = np.nan
        
    ## For date columns
    if len(date_col_idx) > 0:
        print(f'Processing Date columns... {len(date_col_idx)}/{df_og.shape[1]}')
        df_date = df[date_col_idx][:]
        df_date = df_date.apply(pd.to_datetime)  # Ensure all date columns are in datetime format
        
        # Fill missing dates with the mode date
        if df_date.isnull().sum().sum() > 0 and initial_filling != 'None':
            df_date.fillna(df_date.mode().iloc[0], inplace=True)

        kbin = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
        
        date_bins = []
        for col in date_col_idx:
            df_temp = df_date[[col]].dropna()
            df_temp_binned = kbin.fit_transform(df_temp.apply(lambda x: x.astype(int) // 10**9))
            df_temp_binned = pd.DataFrame(df_temp_binned, columns=[f'{col}_BIN{i}' for i in range(5)], index=df_temp.index)
            
            # Merge back to the original dataframe
            df_temp_full = pd.DataFrame(index=df.index)
            df_temp_full = pd.concat([df_temp_full, df_temp_binned], axis=1)
            date_bins.append(df_temp_full)
        
        df_date = pd.concat(date_bins, axis=1)
    
    ## Combine all processed columns
    if len(numerical_col_idx) > 0 and len(categorical_col_idx) > 0 and len(date_col_idx) > 0:
        print(f'[After Processing] Numerical: {df_num.shape[1]} / Categorical: {df_cat.shape[1]} / Date: {df_date.shape[1]}')
        df_final = pd.concat([df_num, df_cat, df_date], axis=1)
    elif len(numerical_col_idx) > 0 and len(categorical_col_idx) > 0:
        print(f'[After Processing] Numerical: {df_num.shape[1]} / Categorical: {df_cat.shape[1]}')
        df_final = pd.concat([df_num, df_cat], axis=1)
    elif len(numerical_col_idx) > 0 and len(date_col_idx) > 0:
        print(f'[After Processing] Numerical: {df_num.shape[1]} / Date: {df_date.shape[1]}')
        df_final = pd.concat([df_num, df_date], axis=1)
    elif len(categorical_col_idx) > 0 and len(date_col_idx) > 0:
        print(f'[After Processing] Categorical: {df_cat.shape[1]} / Date: {df_date.shape[1]}')
        df_final = pd.concat([df_cat, df_date], axis=1)
    elif len(numerical_col_idx) > 0:
        print(f'[After Processing] Numerical: {df_num.shape[1]}')
        df_final = df_num
    elif len(categorical_col_idx) > 0:
        print(f'[After Processing] Categorical: {df_cat.shape[1]}')
        df_final = df_cat
    elif len(date_col_idx) > 0:
        print(f'[After Processing] Date: {df_date.shape[1]}')
        df_final = df_date

    df_final.index = _df.index 
    
    return df_final

# Create a label set
label_set = set(label.PTID.unique())

# bio_2 preprocessing
tmp = bio_1
id_list = []
id_map = {}
for i in range(tmp.shape[0]):
    if pd.notna(tmp.loc[i, 'Phase']):
        if pd.notna(tmp.loc[i, 'RID']):
            _tmp = tmp.loc[i]['Phase'][-1]+'_'+str(tmp.loc[i]['RID'])
            id_list.append(_tmp)
            id_map[_tmp] = tmp.loc[i]['PTID']

ptid_list = []
df = bio_2
key = 'PHASE'
for i in range(df.shape[0]):
    if pd.notna(df.loc[i, key]):
        if pd.notna(df.loc[i, 'RID']):
            _tmp = df.loc[i][key][-1]+'_'+str(df.loc[i]['RID'])
            if _tmp in list(id_map.keys()):
                ptid_list.append(id_map[_tmp])
            else:
                ptid_list.append(np.nan)
        else:
            ptid_list.append(np.nan)
    else:
        ptid_list.append(np.nan)    
df['PTID'] = ptid_list

# List of biospecimen DataFrames
bios = [bio_1, bio_2]

# Convert column names of each DataFrame to uppercase
for i in range(len(bios)):
    bios[i].columns = bios[i].columns.str.upper()

# Update each DataFrame to use the most recent 'UPDATE_STAMP' value
latest_dfs = []
for df in bios:
    df['UPDATE_STAMP'] = pd.to_datetime(df['UPDATE_STAMP'])
    df = df.sort_values(by='UPDATE_STAMP').groupby('PTID').apply(lambda group: group.ffill().bfill().iloc[-1]).reset_index(drop=True)
    latest_dfs.append(df)

# Create a base DataFrame for merging
result_df = latest_dfs[0].copy()

for df in latest_dfs[1:]:
    result_df = pd.merge(result_df, df, on=['PTID'], how='outer', suffixes=('', '_dup'))
    
    # Handle duplicate columns
    for col in df.columns:
        if col != 'PTID' and col != 'UPDATE_STAMP':
            if col + '_dup' in result_df.columns:
                result_df[col] = result_df[col].combine_first(result_df[col + '_dup'])
                result_df.drop(columns=[col + '_dup'], inplace=True)

# Update the 'UPDATE_STAMP' and remove duplicates
update_stamp_cols = [col for col in result_df.columns if 'UPDATE_STAMP' in col]
result_df['UPDATE_STAMP'] = result_df[update_stamp_cols].max(axis=1)
result_df.drop(columns=[col for col in result_df.columns if 'UPDATE_STAMP' in col and col != 'UPDATE_STAMP'], inplace=True)

exclude_set = set(label[np.isnan(label.DIAGNOSIS)].PTID)
label_set = set(label.PTID) - exclude_set
unique_set = [idx for idx in label_set if label[label.PTID == idx].DIAGNOSIS.nunique() == 1]

df_label = label[label['PTID'].isin(unique_set)][['PTID','DIAGNOSIS']].drop_duplicates('PTID')
df_label.reset_index(drop=True, inplace=True)
df_label.to_csv('../label.csv')

cols_1 = set(bios[0].columns)
cols_2 = set(bios[1].columns)

common_cols = cols_1.intersection(cols_2)

# Remove common columns from result_df (except for PTID)
result_df = result_df.drop(columns=[col for col in common_cols if col != 'PTID'])

# Keep only the rows in result_df that have PTID present in label_df and reindex
result_df = result_df[result_df['PTID'].isin(label_set)].reset_index(drop=True)
result_df = result_df.set_index('PTID')

# Save merged csv
final = process_num_cat_cols(result_df)
final = final.replace(True, 1)
final = final.replace(False, -1)
final.to_csv('biospecimen_merged.csv', index=True)
print('Successfully saved merged csv ...!')

# Save merged csv with initial mean imputation
final_2 = process_num_cat_cols(result_df, 'mean')
final_2 = final_2.replace(True, 1)
final_2 = final_2.replace(False, -1)
final_2.to_csv('biospecimen_merged_mean.csv', index=True)
print('Successfully saved merged csv with mean imputation ...!')