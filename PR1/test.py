import numpy as np
from DiagonalCommon import DiagonalCommon
from Common import Common
from DiagonalDifferent import DiagonalDifferent
from Different import Different


# def convtype(data1, data2, data3):
#     conv = input("Type:\n 1) DiagonalCommon\n 2) Common\n 3) DiagonalDifferent\n 4) Different ")
#     if conv == '1':
#         model = DiagonalCommon(data1, data2, data3)
#     elif conv == '2':
#         model = Common(data1, data2, data3)
#     elif conv == '3':
#         model = DiagonalDifferent(data1, data2, data3)
#     elif conv == '4':
#         model = Different(data1, data2, data3)
#     else:
#         print('Invalid')


# dataset = input("Dataset:\n 1) Group 9\n 2) Group 10\n 3) Group 11\n 4) Group 12 ")
# if dataset == '1':
#     data1 = np.loadtxt('Data/linearlySeparableData/{}/Class1.txt'.format(dataset))
#     data2 = np.loadtxt('Data/linearlySeparableData/{}/Class2.txt'.format(dataset))
#     data3 = np.loadtxt('Data/linearlySeparableData/{}/Class3.txt'.format(dataset))
#     convtype(data1, data2, data3)
# elif dataset == '2':
#     data1 = np.loadtxt('Data/linearlySeparableData/{}/Class1.txt'.format(dataset))
#     data2 = np.loadtxt('Data/linearlySeparableData/{}/Class2.txt'.format(dataset))
#     data3 = np.loadtxt('Data/linearlySeparableData/{}/Class3.txt'.format(dataset))
#     convtype(data1, data2, data3)
# elif dataset == '3':
#     data1 = np.loadtxt('Data/linearlySeparableData/{}/Class1.txt'.format(dataset))
#     data2 = np.loadtxt('Data/linearlySeparableData/{}/Class2.txt'.format(dataset))
#     data3 = np.loadtxt('Data/linearlySeparableData/{}/Class3.txt'.format(dataset))
#     convtype(data1, data2, data3)
# elif == dataset == '4':
#     data1 = np.loadtxt('Data/linearlySeparableData/{}/Class1.txt'.format(dataset))
#     data2 = np.loadtxt('Data/linearlySeparableData/{}/Class2.txt'.format(dataset))
#     data3 = np.loadtxt('Data/linearlySeparableData/{}/Class3.txt'.format(dataset))
#     convtype(data1, data2, data3)
# else:
#     print('Invalid')

# Linearly separable data
print("\nLinearly separable data:")
data1 = np.loadtxt('Data/linearlySeparableData/Class1.txt')
data2 = np.loadtxt('Data/linearlySeparableData/Class2.txt')
data3 = np.loadtxt('Data/linearlySeparableData/Class3.txt')

model = DiagonalCommon(data1, data2, data3)
model.result()
model.plot()

model = Common(data1, data2, data3)
model.result()

model = DiagonalDifferent(data1, data2, data3)
model.result()

model = Different(data1, data2, data3)
model.result()
model.plot()


# Non linearly separable data
print("\nNon linearly separable data:")
data = np.loadtxt('Data/nonLinearlySeparableData/data.txt')
data1 = data.head(2446)
data2 = data.tail(2447)

model = DiagonalCommon(data1, data2)
model.result()

model = Common(data1, data2)
model.result()

model = DiagonalDifferent(data1, data2)
model.result()

model = Different(data1, data2)
model.result()
model.plot()


# Real data
print("\nReal data:")
data1 = np.loadtxt('Data/realData/Class1.txt')
data2 = np.loadtxt('Data/realData/Class2.txt')
data3 = np.loadtxt('Data/realData/Class3.txt')

model = DiagonalCommon(data1, data2, data3)
model.result()

model = Common(data1, data2, data3)
model.result()

model = DiagonalDifferent(data1, data2, data3)
model.result()

model = Different(data1, data2, data3)
model.result()
model.plot()
