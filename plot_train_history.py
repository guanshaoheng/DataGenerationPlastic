#!/usr/bin/env python
# encoding: utf-8
"""

@file: plot_train_history.py
@time: 2020/7/18 11:22
@author: Shawn
@email: guanshaoheng@qq.com
@application：

"""

import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt


history_path = r'./new_path_lstm_100_w20'
f = open(os.path.join(history_path, 'loss_history0.txt'), 'r')
datas = f.readlines()

epoch = []
train_loss, validation_loss = [], []
for i in range(1, len(datas)):
    temp_list = datas[i].split(' ')
    temp = temp_list[0].replace('(', '')
    epoch.append(int(temp.replace(',', '')))
    train_loss.append(2.2/2.21*100.*float(temp_list[1].replace(')', '')))
    temp = temp_list[-1].replace(')', '')
    validation_loss.append(1.91/2.21*100.*float(temp.replace('\n', '')))

epoch = numpy.array(epoch)/1e4
x = list(range(0, len(train_loss), 1))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch, [train_loss[i] for i in x], label='Training Loss')
ax.plot(epoch, [validation_loss[i] for i in x], label='Validation Loss')
ax.scatter(epoch[train_loss.index(min(train_loss))], min(train_loss), c='r', zorder=30)
ax.scatter(epoch[validation_loss.index(min(validation_loss))], min(validation_loss), c='b', zorder=30)

plt.annotate(s='Training best %s' % format(min(train_loss), '0.2e'),
             xy=(epoch[train_loss.index(min(train_loss))],
                 min(train_loss),),
             xytext=(epoch[train_loss.index(min(train_loss))] - 30000 / 1e4,
                     min(train_loss) + 1e-4), color="k",
             # fontdict={'fontsize': 12,
             #  'fontweight': 'medium',
             #  'horizontalalignment': 'right',
             #  'fontstyle': 'italic',
             #  'bbox': {'boxstyle': 'round', 'edgecolor': (1., 0.5, 0.5), 'facecolor': (1., 0.8, 0.8)},
             #  },
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="k"),
             fontsize=15
             )

plt.annotate(s='Validation best %s' % format(min(validation_loss), '0.2e'),
             xy=(epoch[validation_loss.index(min(validation_loss))],
                 min(validation_loss),),
             xytext=(epoch[validation_loss.index(min(validation_loss))] - 40000 / 1e4,
                     min(validation_loss) + 5e-4), color="k",
             # fontdict={'fontsize': 12,
             #  'fontweight': 'medium',
             #  'horizontalalignment': 'right',
             #  'fontstyle': 'italic',
             #  'bbox': {'boxstyle': 'round', 'edgecolor': (1., 0.5, 0.5), 'facecolor': (1., 0.8, 0.8)},
             #  },
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="k", ),
             fontsize=15
             )

ax.set_yscale('log')
# plt.show()
plt.legend(fontsize=15, )
plt.xlabel('Training Epoch (x$10^3$)', fontsize=15)
plt.ylabel('Mean square error', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# ax.xaxis.get_major_formatter().set_powerlimits((0, 1), fontsize=15)
plt.locator_params(axis="x", nbins=6)
# plt.grid()
# plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.12)
plt.tight_layout()
name = os.path.join(history_path, 'loss_process.svg')
print(name)
plt.savefig(name)
# plt.show()