import pandas as pd

data = pd.read_hdf("/data/LiYuxiang/new/flow_prediction/data/flow/gantry_flow.hdf")
door_read = pd.read_csv("/data/LiYuxiang/HuaWei/YunNan_Flow/data/door.csv")
door_read = door_read[['ORGNAME', 'GBID']]
door_read_sorted=door_read.sort_values(by='ORGNAME')
# 首先，对 gantryid 进行匹配并赋值
door_relation = data.merge(door_read, left_on='gantryid', right_on='GBID', how='left')
door_relation.rename(columns={'ORGNAME': 'gantryid_ORGNAME'}, inplace=True)
door_read_sorted=door_read_sorted.reset_index(drop=True)
columns_to_keep = ['gantryid', 'gantryname','statisticalhour','vehicleflow','gantryid_ORGNAME']
# 删除不需要的列
door_relation=door_relation[columns_to_keep]
door_relation.dropna(subset=['gantryid_ORGNAME'], inplace=True) #删除gantryid_ORGNAME值为NaN的行 
door_relation.reset_index(drop=True, inplace=True)#重置索引

ew_data=door_relation
ew_data['statisticalhour'] = pd.to_datetime(ew_data['statisticalhour'])
ew_data_2023=ew_data
import pandas as pd
from datetime import datetime, timedelta
df = ew_data_2023

# Convert 'statisticalhour' to datetime
df['statisticalhour'] = pd.to_datetime(df['statisticalhour'])

# Generate a complete datetime range for 2023
start_date = datetime(2023, 1, 1, 0, 0)
end_date = datetime(2023, 12, 31, 23, 0)
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

# 初始化空的DataFrame用于最终的数据汇总
df_complete = pd.DataFrame()

for gantryid in door_read_sorted['GBID'].unique():
    # 根据gantryid过滤数据框
    df_filtered = df[df['gantryid'] == gantryid]

    # 如果有匹配的记录，则继续正常处理
    if not df_filtered.empty:
        df_merged = pd.DataFrame({'statisticalhour': date_range}).merge(df_filtered, on='statisticalhour', how='left')
        gantryname = df_filtered['gantryname'].iloc[0]
        gantryid_ORGNAME = df_filtered['gantryid_ORGNAME'].iloc[0]
    else:
        # 如果没有匹配的记录，创建一个全新的DataFrame，其中只包含日期范围和当前gantryid，其余字段填充为预设值或0
        df_merged = pd.DataFrame({'statisticalhour': date_range})
        gantryname = "未知"  # 或者其他你希望用于填充的值
        gantryid_ORGNAME = "未知"  # 或者其他你希望用于填充的值

    # 填充缺失值
    df_merged['vehicleflow'] = df_merged.get('vehicleflow', pd.Series([0]*len(df_merged)))  # 如果vehicleflow列不存在，则创建并填充0
    df_merged['gantryid'] = gantryid
    df_merged['gantryname'] = gantryname
    df_merged['gantryid_ORGNAME'] = gantryid_ORGNAME

    # 将处理后的DataFrame添加到最终的汇总DataFrame中
    df_complete = pd.concat([df_complete, df_merged], ignore_index=True)

# 显示最终数据框的头部和尾部
df_complete.head(), df_complete.tail(), df_complete.shape
import numpy as np
Imputation_data=df_complete
N = len(Imputation_data['gantryid'].unique())
Imputation_data_door = Imputation_data['vehicleflow'].values.reshape(N, -1, 1)
print("Imputation0_data_door.shape",Imputation_data_door.shape)
np.save('src/Imputation0_data_door.npy', Imputation_data_door)