import pred_soh
import matplotlib.pyplot as plt
import pandas as pd
soh_seq=(pd.read_csv(r'C:\Users\ycc\Desktop\SOH\car7_LSTM.csv')).values

data_y = soh_seq[1:,0]
a = soh_seq[:,1]#读取第二列
b = list(a)
train_size = 173




fig,ax= plt.subplots(figsize=(4,4),facecolor='w',dpi=600) 
ax.set_ylim(0,1.2)
plt.scatter(b[-3], pred_y[train_size + 63], marker='*', s=100, zorder=30)
print('after 180days:', pred_y[train_size + 63])
plt.plot(b[train_size-3:-2], pred_y[train_size-3:], 'r', label='pred', linewidth=1.5, zorder=20)
plt.plot(b[:train_size-1],data_y[:train_size-1], 'b', label='real', alpha=0.6, linewidth=1.5)
# plt.plot([train_size, train_size], [data_y[train_size-1]-0.008, data_y[train_size-1]+0.008], linestyle='--', color='k', label='train | pred')
plt.title('Car_{}'.format('7'), fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Timestamp', fontsize=10)
plt.ylabel('RUL', fontsize=10)
plt.legend(loc='best')
 # plt.savefig('lstm_reg.png')