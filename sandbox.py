import torch

dct1 = {}
dct2 = {}
with open('tmp', 'r') as file:
    for line in file:
        a,b,c = line.split()
        dct1[a] = (b,c)
with open('orig', 'r') as file:
    for line in file:
        a,b,c = line.split()
        dct2[a] = (b,c)
time_sum1 = 0
expl_size_sum1 = 0
time_sum2 = 0
expl_size_sum2 = 0
ratio = 0
for k in dct1:
    time_sum1 += float(dct1[k][1])
    expl_size_sum1 += float(dct1[k][0])
    time_sum2 += float(dct2[k][1])
    expl_size_sum2 += float(dct2[k][0])
    if float(dct2[k][1]) != 0:
        ratio += float(dct1[k][1])/float(dct2[k][1])
        print(ratio, float(dct1[k][1]), float(dct2[k][1]))
print(time_sum1/17, expl_size_sum1/17)
print(time_sum2/17, expl_size_sum2/17)
print(ratio/17)


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