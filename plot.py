import matplotlib.pyplot as plt
import numpy as np

# DD = [
#     [77.64,78.38,77.87,77.75,77.63],
#     [77.21,77.98,77.81,77.69,77.57],
#     [78.07,78.63,78.83,78.86,78.51],
#     [77.90,78.42,78.41,78.42,78.23]
# ]
#
# PROTEINS = [
#     [74.26,75.33,75.07,75.24,75.39],
#     [74.37,75.23,75.05,74.82,75.33],
#     [74.17,75.22,75.35,75.58,75.16],
#     [74.20,75.28,75.07,75.15,75.06]
# ]
#
# x = [1, 2, 3, 4, 5]
#
# fig = plt.figure()
# plt.subplot(2,1,1)
# plt.grid()
# plt.axis([0.95, 5.05, 74, 80])
# plt.plot(x, DD[2], "o-", label=r'MHAG$_g$', linewidth=2.0)
# plt.plot(x, DD[3], "o-", label=r'AMHAG$_g$', linewidth=2.0)
# plt.plot(x, DD[0], "o-", label=r'MHAG$_h$', linewidth=2.0)
# plt.plot(x, DD[1], "o-", label=r'AMHAG$_h$', linewidth=2.0)
# plt.xticks(x, ('1', '2', '4', '8', '16'), fontsize=12)
# plt.yticks(np.arange(74, 80, 0.5), fontsize=10)
# plt.xlabel('Head number', fontsize=12, labelpad=-2)
# plt.ylabel('Avg accuracyv (DD)', fontsize=12, labelpad=-0)
# # plt.title('Accuracy', fontsize=20)
# plt.legend(ncol=4, fontsize=10.8)
# plt.subplot(2,1,2)
# plt.grid()
# plt.axis([0.95, 5.05, 72, 78])
# plt.plot(x, PROTEINS[2], "o-", label=r'MHAG$_g$', linewidth=2.0)
# plt.plot(x, PROTEINS[3], "o-", label=r'AMHAG$_g$', linewidth=2.0)
# plt.plot(x, PROTEINS[0], "o-", label=r'MHAG$_h$', linewidth=2.0)
# plt.plot(x, PROTEINS[1], "o-", label=r'AMHAG$_h$', linewidth=2.0)
# plt.xticks(x, ('1', '2', '4', '8', '16'), fontsize=12)
# plt.yticks(np.arange(72, 78, 0.5), fontsize=10)
# plt.xlabel('Head number', fontsize=12, labelpad=-2)
# plt.ylabel('Avg accuracy (PROTEINS)', fontsize=12, labelpad=-0)
# # plt.title('F1', fontsize=20)
# plt.legend(ncol=4, fontsize=10.8)
# plt.tight_layout(h_pad=-1.5)
# plt.show()

# d = [
#     [0, 18.97, 9.72, 64.80, 65.25, 55.53, 79.85],
#     [16.72,0,77.22,25.10,18.21,14.32,82.76],
#     [21.21,81.53,0,28.41,22.65,17.44,81.37],
#     [40.03,9.38,5.89,37.74,0,28.36,52.05],
#     [29.58,7.10,5.68,27.14,0,16.15,48.98],
#     [19.06,4.4,4.69,12.36,14.75,0,44.67]
# ]
#
# new_d = [[],[],[],[],[],[],[]]
#
# for dataset in d:
#     for i, s in enumerate(dataset):
#         new_d[i].append(s)
# x = np.array([1, 2, 3, 4, 5, 6])
# bar_width = 0.1
# labels = ['SQuAD', 'CNN', 'DailyMail', 'NewsQA', 'CoQA', 'DROP', 'Self']
#
# fig = plt.figure()
# plt.grid()
# plt.axis([0.5, 6.5, 0, 90])
# for i in range(7):
#     plt.bar(x - 0.35 + bar_width * i, new_d[i], bar_width, label=labels[i])
# plt.xticks(x, ('SQuAD', 'CNN', 'DailyMail', 'NewsQA', 'CoQA', 'DROP'), fontsize=15)
# plt.yticks(np.arange(0, 100, 10), fontsize=15)
# plt.xlabel('Target dataset', fontsize=15, labelpad=-2)
# plt.ylabel('Accuracy / %', fontsize=15, labelpad=-0)
# plt.legend(ncol=2, fontsize=15)
# plt.show()

data = [
    [18.97, 79.86, 78.59, 79.85],
    [81.53, 84.26, 83.40, 81.37],
    [9.38, 48.37, 48.95,52.05],
    [29.58,52.38,50.77,48.98],
    [12.36,47.36,45.06,44.67],
    [17.44,80.34,80.78,81.37],
    [64.80, 80.17, 78.87, 79.85],
    [22.65, 76.87, 78.12, 81.37]
]
labels = ['Zero-shot', 'CASe','CASe+E','Self']

xt = ['SQuAD->CNN', 'DailyMail->CNN', 'CNN->NewsQA', 'SQuAD->CoQA', 'NewsQA->DROP', 'DROP->DailyMail',
          'NewsQA->SQuAD', 'CoQA->DailyMail']
new_d = [[],[],[],[]]
for dataset in data:
    for i, s in enumerate(dataset):
        new_d[i].append(s)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
bar_width = 0.2

fig = plt.figure()
plt.grid()
plt.axis([0.5, 8.5, 0, 90])
for i in range(4):
    plt.bar(x - 0.3 + bar_width * i, new_d[i], bar_width, label=labels[i])
plt.xticks(x, xt, fontsize=14)
plt.yticks(np.arange(0, 100, 10), fontsize=15)
# plt.xlabel('Target dataset', fontsize=15, labelpad=-2)
plt.ylabel('Accuracy / %', fontsize=15, labelpad=-0)
plt.legend(ncol=2, fontsize=15)
plt.show()



