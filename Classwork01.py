import numpy as np

x = [29,28,34,31,25]
y = [77,62,93,84,59]
diffX = []
diffY = []
listSxy = []
AvgX = sum(x)/len(x)
AvgY = sum(y)/len(y)

for i in x:
    xx = i - AvgX
    diffX.append(xx)
for w in y:
    yy = w - AvgY
    diffY.append(yy)

preSxx = [Sx**2 for Sx in diffX]
preSyy = [Sy**2 for Sy in diffY]
Sxx = sum(preSxx)
Syy = sum(preSyy)

listSxy = [diffX[i] * diffY[i] for i in range(min(len(diffX), len(diffY)))]
print(listSxy)


Sxy = sum(listSxy)
a = Sxy/Sxx
b = AvgY-AvgX*a

print(f'the regression equation is y = {a:.3f}x + ({b:.3f})')






