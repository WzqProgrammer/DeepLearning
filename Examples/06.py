num = int(input("请输入数字："))

isHuiwen = True
num_list = []
y = num

while(y>0):
    temp = y % 10
    num_list.append(temp)
    y = int(y/10)

print("列表：" + str(num_list))

for i in range(len(num_list)):
    if num_list[i] != num_list[len(num_list)-i-1]:
        isHuiwen = False

if isHuiwen == True:
    print("是回文数")
else:
    print("不是回文数")
print(num,num_list)