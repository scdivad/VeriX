import torch

dct1 = {}
dct2 = {}
with open('midpoint/my', 'r') as file:
    for line in file:
        a,b,c = line.split()
        dct1[a] = (b,c)
with open('midpoint/baseline', 'r') as file:
    for line in file:
        a,b,c = line.split()
        dct2[a] = (b,c)
time_sum1 = 0
expl_size_sum1 = 0
time_sum2 = 0
expl_size_sum2 = 0
ratio_explanation = 0
ratio_time = 0
cnt_nnz = 0
point = dct1 if len(dct1) < len(dct2) else dct2
for k in point:
    if (float(dct1[k][1]) != 0 and float(dct2[k][1]) != 0):
        time_sum1 += float(dct1[k][0])
        expl_size_sum1 += float(dct1[k][1])
        time_sum2 += float(dct2[k][0])
        expl_size_sum2 += float(dct2[k][1])
        ratio_time += float(dct1[k][0])/float(dct2[k][0])
        ratio_explanation += float(dct1[k][1])/float(dct2[k][1])
        cnt_nnz += 1

print("average ratio explanation: ", ratio_explanation/cnt_nnz)
print("average ratio time: ", ratio_time/cnt_nnz)


# # Open the file and iterate over each line
# with open('my.txt', 'r') as file:
#     sum = 0
#     for line in file:
#         if line.startswith('Time elapsed'):
#             # get index of last space
#             idx = line.rfind(' ')
#             # get the substring after the last space
#             sum += float(line[idx+1:])
#     print(sum)

# with open('orig.txt', 'r') as file:
#     sum = 0
#     for line in file:
#         if line.startswith('Time elapsed'):
#             # get index of last space
#             idx = line.rfind(' ')
#             # get the substring after the last space
#             sum += float(line[idx+1:])
#     print(sum)