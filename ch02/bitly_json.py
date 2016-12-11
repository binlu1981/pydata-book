import json
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
print open(path).readline()
"""
{ "a": "Mozilla\/5.0 (Windows NT 6.1; WOW64) AppleWebKit\/535.11 (KHTML, like Gecko) Chrome\/17.0.963.78 Safari\/535.11", "c": "US", "nk": 1, "tz": "America\/New_York", "gr": "MA", "g": "A6qOVH", "h": "wfLQtf", "l": "orofrog", "al": "en-US,en;q=0.8", "hh": "1.usa.gov", "r": "http:\/\/www.facebook.com\/l\/7AQEFzjSi\/1.usa.gov\/wfLQtf", "u": "http:\/\/www.ncbi.nlm.nih.gov\/pubmed\/22415991", "t": 1331923247, "hc": 1331822918, "cy": "Danvers", "ll": [ 42.576698, -70.954903 ] }
"""
records = [json.loads(line) for line in open(path)]
print records[0]['tz']
"""
America/New_York
"""
#time_zone = [rec['tz'] for rec in records]
#KeyError: 'tz' (some rec lost key 'tz')
time_zone = [rec['tz'] for rec in records if 'tz' in rec]
print time_zone[:10]
"""
[u'America/New_York', u'America/Denver', u'America/New_York', u'America/Sao_Paulo', u'America/New_York', u'America/New_York', u'Europe/Warsaw', u'', u'', u'']
"""
def get_counts(lst):
    count_dict = {}
    for x in lst:
        if x in count_dict:
            count_dict[x] += 1
        else:
            count_dict[x] = 1
    return count_dict

def get_counts1(lst):
    count_dict = {}
    for x in lst:
        count_dict[x] = count_dict.get(x,0) + 1
    return count_dict
    
from collections import defaultdict
def get_counts2(lst):
    count_dict = defaultdict(int)
    for x in lst:
        count_dict[x] += 1
    return count_dict
    
def top_counts(count_dict,n=10):
    v_k_list = sorted([(v,k) for k, v in count_dict.items()])
    return v_k_list[-n:]

def top_counts1(count_dict,n=10):
    v_k_list = sorted(count_dict.items(),key = lambda x: (x[1],x[0])) 
    return v_k_list[-n:]
counts_dict = get_counts(time_zone)
print len(time_zone) #3440
counts_dict = get_counts1(time_zone)
print top_counts1(counts_dict,10)
"""
[(33, u'America/Sao_Paulo'), (35, u'Europe/Madrid'), (36, u'Pacific/Honolulu'), (37, u'Asia/Tokyo'), (74, u'Europe/London'), (191, u'America/Denver'), (382, u'America/Los_Angeles'), (400, u'America/Chicago'), (521, u''), (1251, u'America/New_York')]
"""
  