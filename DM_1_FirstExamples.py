import json
from collections import Counter
from pandas import DataFrame, Series
import pandas as pd


def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]





path = 'usa_gov.txt'
#open(path).readline()
records = [json.loads(line) for line in open(path)]
#print(records[0]['tz'])

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
#print(time_zones[:10])

counts = get_counts(time_zones)

#print(counts)
#print(counts['America/New_York'])
# print(len(time_zones))


#print(top_counts(counts))

counts = Counter(time_zones)
#print(counts.most_common(10))

frame = DataFrame(records)
#print(frame['tz'][:10])

tz_counts = frame['tz'].value_counts()
#print(tz_counts[:10])

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])

tz_counts[:10].plot(kind='barh', rot=0)









































































