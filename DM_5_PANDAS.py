import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd

#np.random.seed(19680801)

#mu, sigma = 100, 15
#x = np.random.randn(10000)

# the histogram of the data
#n, bins, patches = plt.hist(x, 50, facecolor='g')


#plt.xlabel('Smarts')
#plt.ylabel('Probability')
#plt.title('Histogram of IQ')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
#plt.grid(True)
#plt.show()

obj = Series([4, 7, -5, 3])

#print(obj)
#print(obj.values)
#print(obj.index)

obj = Series([4, 7, -5, 2], index=['d', 'b', 'a', 'c'])

#print(obj)

#print(obj.index)

#print(obj['a'])

#print(obj[obj > 0])

#print(obj * 2)

#print(np.exp(obj))

#print('b' in obj)
#print('e' in obj)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}

obj = Series(sdata)

#print(obj)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj = Series(sdata, index=states)

#print(obj)

#print(pd.isnull(obj))

#print(pd.notnull(obj))

#print(obj.isnull())

obj.name = 'population'
obj.index.name = 'state'

#print(obj)

#obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']

#print(obj)


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.6]}

frame = DataFrame(data)

#print(frame)

#print(DataFrame(data, columns=['year', 'state', 'pop']))

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])

#print(frame2)

#print(frame2.columns)

#print(frame2['state'])

#print(frame2.ix['three'])

frame2['debt'] = 16.5

#print(frame2)

frame2['debt'] = np.arange(5)

#print(frame2)

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])

frame2['eastern'] = frame2.state == 'Ohio'

#print(frame2)

del frame2['eastern']

#print(frame2.columns)

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = DataFrame(pop)

#print(frame3)

#print(frame3.T)

#print(DataFrame(pop, index=[2001, 2002, 2003]))


pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}

#print(DataFrame(pdata))

frame3.index.name = 'year'
frame3.columns.namme = 'state'

#print(frame3.values)

#print(frame2.valuesi)

obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index

#print(index)

index = pd.Index(np.arange(3))

obj2 = Series([1.5, -2.5, 0], index=index)

#print(obj2.index is index)

#print(frame3)

#print('Ohio' in frame3.columns)

#print(2003 in frame3.index)

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

#print(obj2)

obj = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)

#print(obj)

obj3 = Series(['blue', 'purple', 'yellow'], index=(0, 2, 4))

obj3 = obj3.reindex(range(6), method='ffill')

#print(obj3)

frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])

#print(frame)

frame2 = frame.reindex(['a', 'b', 'c', 'd'])

#print(frame2)

states = ['Texas', 'Utah', 'California']

frame = frame.reindex(columns=states)

#print(frame)

frame = frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=states)

#print(frame)

#print(frame.ix[['a', 'b', 'c', 'd'], states])

obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')

#print(new_obj)

data = DataFrame(np.arange(16).reshape(4, 4),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

#print(data.drop(['Colorado', 'Ohio']))

#print(data.drop('two', axis=1))


obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])

#print(obj[['b', 'a', 'd']])

#print(obj['b':'c'])

obj['b':'c'] = 5

#print(obj)

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])

#print(data)

#print(data['two'])

#print(data[data['three'] > 5])

#print(data < 5)


#print(data.ix['Colorado', ['two', 'three']])


#print(data.ix[['Colorado', 'Utah'], [3, 0, 1]])
























































