import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

print('=========1===========')

x = [63.4, 80.4, 127.8, 87.1, 102.0, 77.8, 61.4, 89.7, 119.3, 70.6,
	80.5, 101.0, 113.8, 110.3, 55.1, 101.9, 89.8, 95.7, 90.2, 65.4,
	104.1, 82.0, 82.1, 104.6, 94.6, 61.0, 94.7, 66.0, 79.1, 73.7,
	99.6, 95.2, 120.3, 89.9, 105.1, 85.8, 112.2, 102.5, 96.2, 77.0,
	116.3, 82.0, 81.4, 80.4, 86.2, 57.8, 107.3, 61.6, 80.4, 113.7,
	126.8, 88.2, 89.9, 96.5, 83.3, 113.2, 111.6, 108.6, 86.0, 82.9]
mx = 90.0
sx = 20.0
lambda_const = 0.0111
x.sort()
print('X = ', x)

ranges = [mx - 3 * sx, mx + 3 * sx] # D(f) = (30; 150)
f_x = []
for i, num in enumerate(range(len(x))):
	f_x.append((i+1)*0.01167)
print(f_x)

def exp_dis(lamb_da, calc):
	dis_calc = 1 - np.exp(-lamb_da*calc)
	#dis_R = stats.expon.cdf(calc,1/lamb_da)
	dis_R = 0
	if calc:
		return dis_calc
	else:
		return dis_R

expdist = []
exp = 0
for num in x:
	y = exp_dis(lambda_const, num)
	expdist.append(y);
	exp += 0.1;


def getDiffValue(a, b):
	diffFunc = []
	for i in range(len(x)):
		diffFunc.append(abs(a[i] - b[i]));
	return diffFunc;


diffFuncExp = getDiffValue(expdist, f_x)
maxIndex = diffFuncExp.index(max(diffFuncExp))
maxDiff = x[maxIndex]

gaussDistr =[]
for num in x:
	y = stats.norm(mx, sx).cdf(num)
	gaussDistr.append(y)

diffFuncNormStd = getDiffValue(gaussDistr, f_x)
maxNormStdIndex = diffFuncNormStd.index(max(diffFuncNormStd))
maxNormStdDiff = x[maxNormStdIndex]

maxDiffNormStd = max(diffFuncNormStd)
print('Max diff between Standard Normal DF and Empirical DF = ', maxDiffNormStd)

alpha = 0.1
lambdaCritical = 1.224
lambdaCritery = maxDiffNormStd*np.sqrt(len(x))
print('Lambda Critery =', lambdaCritery)

def getpk(alpha):
	pk = [[0.2, 1.073], [0.1, 1.224], [0.05, 1.358], [0.02, 1.52], [0.01, 1.667], [0.001, 1.95]]
	for num in pk:
		if (num[0] == alpha):
			return alpha;


def checkHypothesis(lambd, alpha):
	pk = getpk(alpha)
	if (lambd <= pk):
		return 'H0', lambd, '<=', pk, 'Hypothesis is accepted'
	else:
		return 'H1', lambd, '>', pk, 'Hypothesis is not accepted'


print('1.1: ', checkHypothesis(lambdaCritery, alpha))

plt.plot([maxNormStdDiff, maxNormStdDiff], [gaussDistr[maxNormStdIndex], f_x[maxNormStdIndex]], linestyle='--')
plt.title('Check of Standard Normal Distribution Hypothesis')
plt.plot(x, f_x, drawstyle='steps', label='Empirical Distribution Function')
plt.plot(x, gaussDistr, label='Theoretical Distribution Function')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

#=============1.2===============

alpha = 0.05

ax = plt.subplot()
plt.title('Check of Exponential Hypothesis')
plt.plot(x, expdist, label='Theoretical Distribution Function')
plt.plot([maxDiff, maxDiff], [f_x[maxIndex], expdist[maxIndex]], color='red', linestyle='--', linewidth='1.5')
plt.plot(x, f_x, drawstyle='steps', label='Empirical Distribution Function');
plt.grid(True)
plt.legend(loc='upper left')
plt.show()

maxDiffExp = max(diffFuncExp)
lambdaCritery = maxDiffExp*np.sqrt(len(x))
print('Max diff between Exponential DF and Empirical DF = ', maxDiffExp)
print('1.2: ', checkHypothesis(lambdaCritery, alpha))

#======2======
print('=========2===========')

x = [3, 2, 1, 1, 4, 2, 3, 1, 2, 2,
	 1, 2, 1, 1, 3, 2, 3, 1, 2, 2,
	 6, 4, 1, 1, 5, 5, 3, 1, 7, 1,
	 1, 4, 1, 1, 3, 5, 1, 10, 13, 3,
	 1, 2, 1, 5, 5, 5, 3, 5, 1, 2,
	 2, 3, 2, 3, 8, 3, 3, 2, 2, 1,
	 3, 2, 1, 1, 4, 7, 4, 6, 1, 2,
	 1, 4, 2, 4, 1, 7, 4, 5, 1, 2,
	 3, 6, 3, 2, 1, 7, 1, 1, 2, 8,
	 1, 2, 8, 2, 8, 2, 8, 2, 2, 9,
	 2, 2, 6, 1, 4, 1, 3, 2, 3, 2]

xuniq = []
for num in x:
	if(num not in xuniq):
		xuniq.append(num)

quantityUniq = []
for num in xuniq:
	quantityUniq.append(x.count(num))

xnuniq = []
for xint, nint in zip(xuniq, quantityUniq):
	xnuniq.append([xint, nint])
nxuniq = sorted(xnuniq, key=lambda x: x[0])


#print('n:', quantityUniq)
#print('x:', xuniq)
#print('quanx = ', len(x))
#print('qquantit', len(quantityUniq))
print('(x,n):', nxuniq)

mx = 0
for num in x:
	mx += num

mx /= len(x)
p = 1/mx
q = 1 - p

xnpuniq = []
for xuniq, n in nxuniq:
	prob = p*q**(xuniq - 1)
	xnpuniq.append([xuniq, n, prob])

print('mx* =', mx)
print('p* =', p)
print('q* =', q)
print('(x, n, p):', xnpuniq)
print('Count of x:', len(xnpuniq))

df = len(xnpuniq) - 1 -1 #Degree of freedom

alpha = 0.05
xsqr = 0
for i in range(df):
	xuniq, n, p = xnpuniq[i]
	xsqr += ((n-len(x)*p)**2)/(len(x)*p)
xsqrCritical = 19.7
print('X^2 =', xsqr)

def checkXHypothesis(xsqrCritical, xsqr):
	if (xsqr <= xsqrCritical):
		return 'H0', xsqr, '<=', xsqrCritical, 'Hypothesis is accepted'
	else:
		return 'H1', xsqr, '>', xsqrCritical, 'Hypothesis is not accepted'


print('2:', checkXHypothesis(xsqrCritical, xsqr))

#======3======
print('=========3.1===========')

x = [[120.37, 86.66], [57.33, 58.65], [93.43, 79.50], [90.92, 80.97], [76.36, 62.83], [95.13, 65.46], [79.76, 66.52],
	 [55.71, 53.67], [74.59, 56.88], [82.72, 72.52], [135.83, 83.33], [31.59, 51.60], [91.21, 71.18],
	 [39.91, 50.62], [61.46, 67.78], [97.03, 61.62], [100.37, 86.36], [64.07, 71.40], [54.87, 49.60]]

x = sorted(x, key=lambda x: x[0])
print(x)
print('n =', len(x))

mxNotBias = 0
myNotBias = 0
for xnum, ynum in x:
	mxNotBias += xnum
	myNotBias += ynum

mxNotBias *= (1/len(x))
myNotBias *= (1/len(x))
print('mx* not bias =', mxNotBias)
print('my* not bias =', myNotBias)

dx = 0
kxy = 0

for xnum, ynum in x:
	dx += (xnum - mxNotBias)**2
	kxy += (xnum - mxNotBias)*(ynum - myNotBias)

dx *= 1/(len(x))
kxy *= 1/(len(x))

print('Dx* =', dx)
print('Kxy* =', kxy)

b1 = kxy/dx
b0 = myNotBias - b1*mxNotBias

print('b0 =', b0, ';b1 =', b1)


xtemp = []
ytemp = []

for xlocal, ylocal in x:
	plt.plot(xlocal, ylocal, linestyle='', marker='o', color='red')
	xtemp.append(xlocal)
	ytemp.append(ylocal)

print(xtemp)
step = 5
temp = np.floor(xtemp[0])
xregr = []
yregr = []
i = 0
#for num in range(len(xtemp) + 1):
while(temp <= np.ceil(xtemp[len(xtemp) - 1])):
	xregr.append(temp)
	yregr.append(b0 + b1*temp)
	i = i + 1
	temp += step


print(xregr)

plt.plot(xregr, yregr, color='blue', label='Regression')
plt.legend()
plt.grid(True)
plt.show()

Sxx = 0
Sxy = 0

for xregrlocal, yregrlocal in zip(xregr, yregr):
	Sxx += (xregrlocal - mxNotBias)**2
	Sxy += (xregrlocal - mxNotBias)*(yregrlocal - myNotBias)

Se = 0

for yregrlocal, ylocal in zip(yregr, ytemp):
	Se += (yregrlocal - ylocal)**2

sigmaEpsSqr = Se/(len(x))
sigmaB0 = np.sqrt(sigmaEpsSqr*((1/len(x)) + (mxNotBias**2)/Sxx))
sigmaB1 = np.sqrt(sigmaEpsSqr/Sxx)

print('Sxx =', Sxx)
print('Sxy =', Sxy)
print('Se =', Se)
print('Sigma Epsilon Square =', sigmaEpsSqr)
print('Sigma(b0) =', sigmaB0)
print('Sigma(b1) =', sigmaB1)

beta00 = 40
beta10 = 0.4
#=====CheckHypothesis=====

print('=========3.1===========')
#===3.1===
t = (b1 - beta10)/sigmaB1
tCritical = 2.90
t2Critical = tCritical
t1Critical = - t2Critical

print('t =', t)
print('t1 critical =', t1Critical)
print('t2 critical =', t2Critical)

print('t =', t, '; t 1 critical =', t1Critical, '; t 2 Critical =', t2Critical)

def checkXHypothesis(t, t1Critical, t2Critical):
	if ((t1Critical < t) and (t2Critical > t)):
		return 'H0', t1Critical ,'<', t, '<', t2Critical, 'Hypothesis is accepted'
	else:
		return 'H1', 'Hypothesis is not accepted'


print(checkXHypothesis(t, t1Critical, t2Critical))

#===3.2===
print('=========3.2===========')
t = (b1 - beta10)/sigmaB1
tCritical = 2.11
t2Critical = tCritical
t1Critical = - t2Critical

print('t =', t)
print('t1 critical =', t1Critical)
print('t2 critical =', t2Critical)

print('t =', t, '; t 1 critical =', t1Critical, '; t 2 Critical =', t2Critical)

def checkXH2ypothesis(t, t1Critical, t2Critical):
	if ((t1Critical < t) and (t2Critical > t)):
		return 'H0', t1Critical,'<', t, '<', t2Critical, 'Hypothesis is accepted'
	else:
		return 'H1', 'Hypothesis is not accepted'

print(checkXH2ypothesis(t, t1Critical, t2Critical))

#=====3.3======
print('=========3.3===========')


tCritical = 2.11
print('t =', t)
print('t critical =', tCritical)

if(t < tCritical):
	print('H0:', t, '<', tCritical, ' Hypothesis is accepted')
else:
	print('H1:', t, '>=', tCritical, ' Hypothesis is not accepted')

#===3.4===
print('=========3.4===========')

t = (b0 - beta00)/sigmaB0

tCritical = 2.90
t2Critical = tCritical
t1Critical = -t2Critical

print('t =', t)
print('t1 critical =', t1Critical)
print('t2 critical =', t2Critical)

if((t > t1Critical) and (t < t2Critical)):
	print('H1:', t1Critical ,'<', t, '<', t2Critical, ' Hypothesis is not accepted')
else:
	print('H0:', 'Hypothesis is accepted')

#===3.5===
print('=========3.5===========')

t= b1/sigmaB1

#tCritical = 2.57 # Исправить в предыдущих
t2Critical = 2.57
t1Critical = -t2Critical

print('t =', t)
print('t1 critical =', t1Critical)
print('t2 critical =', t2Critical)

if((t > t1Critical) and (t < t2Critical)):
	print('H0:', t1Critical, '<', t, '<', t2Critical, ' Hypothesis is accepted')
else:
	print('H1:', 'Hypothesis not is accepted')

#===4===

print('=========4===========')

thalfalpha = 2.1098 #t a/2,(n-2) по таблице Стьюдента

Ib1 = [(b1 - thalfalpha*sigmaB1), (b1 + thalfalpha*sigmaB1)]
Ib0 = [(b0 - thalfalpha*sigmaB0), (b0 + thalfalpha*sigmaB0)]

print('Ib0 = ', Ib0)
print('Ib1 = ', Ib1)

#===5===

print('=========5===========')

thalfalpha = 2.5669 #t a/2,(n-2) по таблице Стьюдента

myhi = []
mylo = []

for xlocal, myxlocal in zip(xregr, yregr):
	mylo.append(myxlocal - thalfalpha*np.sqrt(sigmaEpsSqr*((1/len(xregr)) + ((xlocal - mxNotBias)**2)/(Sxx))))
	myhi.append(myxlocal + thalfalpha * np.sqrt(sigmaEpsSqr * ((1 / len(xregr)) + ((xlocal - mxNotBias) ** 2) / (Sxx))))

bx = plt.subplot()
print('Lower confidence limit of expected value =',mylo)
print('Upper confidence limit of expected value =',myhi)
plt.plot(xtemp, ytemp, color='black', linestyle='', marker='o')
plt.plot(xregr, yregr, color='blue', label='Regression')
plt.plot(xregr, myhi, color='green', label='Upper confidence limit of expected value')
plt.plot(xregr, mylo, color='red', label='Lower confidence limit of expected value')
plt.legend()
plt.grid(True)
plt.show()

print('T a/2, (n-2) =', thalfalpha)

#===7===
print('======7======')
thalfalpha = 2.1098

yhi = []
ylo = []

for xlocal, myxlocal in zip(xregr, yregr):
	ylo.append(myxlocal - thalfalpha*np.sqrt(sigmaEpsSqr*(1 + (1/len(xregr)) + ((xlocal - mxNotBias)**2)/(Sxx))))
	yhi.append(myxlocal + thalfalpha * np.sqrt(sigmaEpsSqr * (1 + (1 / len(xregr)) + ((xlocal - mxNotBias) ** 2) / (Sxx))))

cx = plt.subplot()
print('Lower confidence limit =', ylo)
print('Upper confidence limit =', yhi)
plt.plot(xtemp, ytemp, color='black', linestyle='', marker='o')
plt.plot(xregr, yregr, color='blue', label='Regression')
plt.plot(xregr, yhi, color='purple', label='Upper confidence limit')
plt.plot(xregr, ylo, color='pink', label='Lower confidence limit')
plt.plot(xregr, myhi, color='green', label='Upper confidence limit of expected value')
plt.plot(xregr, mylo, color='red', label='Lower confidence limit of expected value')
plt.legend()
plt.grid(True)
plt.show()
