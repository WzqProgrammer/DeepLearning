import numpy as np

# num = int(input("请输入数字："))
#
# isHuiwen = True
# num_list = []
# y = num
#
# while(y>0):
#     temp = y % 10
#     num_list.append(temp)
#     y = int(y/10)
#
# print("列表：" + str(num_list))
#
# for i in range(len(num_list)):
#     if num_list[i] != num_list[len(num_list)-i-1]:
#         isHuiwen = False
#
# if isHuiwen == True:
#     print("是回文数")
# else:
#     print("不是回文数")
# print(num,num_list)

# def find_target(num_list, target):
#     my_list = []
#     if len(num_list)==0:
#         print("输入列表为空")
#         return my_list
#
#     for i in range(len(num_list)-1):
#         for j in range(i+1, len(num_list)):
#             if num_list[i] + num_list[j] == target:
#                 if i not in my_list:
#                     my_list.append(i)
#                 if j not in my_list:
#                     my_list.append(j)
#
#     return my_list
#
# target_list = find_target([1, 2, 3, 4, 2, 5, 1], 4)
# print(target_list)

my_list = [1, 2, 3, 4, 5]
my_list2 = [6, 7, 8, 9, 10, 2]
labels = np.zeros((1, 6))
labels[np.arange(len(labels)), 2] = 1
print(labels)