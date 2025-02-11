import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import re

file_path = '/home/cuitianyu/cuitianyu/data/bytedance/software/input/1.txt'

with open(file_path, 'r') as f:
    lines = f.readlines()

timestamps = []
log_entries = []

for line in lines:
    match = re.match(r'^\[\s*(\d+\.\d+)\]', line)  
    if match:
        timestamp = float(match.group(1)) 
        timestamps.append(timestamp)
        log_entries.append(line.strip())

df = pd.DataFrame({
    'timestamp': timestamps,
    'log_entry': log_entries
})

db = DBSCAN(eps=0.5, min_samples=1) 
df['cluster'] = db.fit_predict(df[['timestamp']])

max_cluster = df['cluster'].max() 
max_cluster_logs = df[df['cluster'] == max_cluster] 

print(f"\n聚类编号最大的一组聚类（cluster {max_cluster}）中的日志内容：")
for log in max_cluster_logs['log_entry']:
    print(log)
