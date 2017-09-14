import pandas as pd
from pandas import Series,DataFrame
#from pandas_datareader import data, web
"""
#!cat ./ex1.csv
df = pd.read_csv('./ex1.csv',index_col='message')
print(df)

print(list(open('ex3.txt')))
print(open('ex3.txt').read())

result = pd.read_table('./ex3.txt',sep='\s+')
print(result)

from lxml.html import parse
import requests

resp = requests.get(url='http://finance.yahoo.com/q/op?s=AAPL+oPTIONS')
print(resp.text)

from lxml import objectify

import json

#data = json.loads(resp.text)
#print(data)

import sqlite3


#con.execute(query)
#con.commit()

"""
"""
import os

print(os.path.basename('./txt'))

a ={"c":"adfdfed","b":"trgrg"}
print(list(a.values()))

for k in list(a.keys()):
    if "c" in k:
        del a[k]
print(a)

import re
import glob

for file in glob.glob("./*rst"):
    with open(file) as f:
        records = f.read()
        match_tree= re.search(r"Ancestral\sreconstruction\sby\sAAML\.\n+.+;\n+(.+);\n.+\n+tree\swith\snode\slabels\sfor\sRod\sPage",records,re.S)
        tree = match_tree.group(1).strip()
        print(tree)


with open("convergeprob_salamander.prob.txt") as f:
    CONVERG2_out = f.read()
    exp_para = re.search(r"Expected\snumber\sof\sparallel-change\ssites\s*=\s*(\d*\.?\d*)",CONVERG2_out,re.S).group(1)
    obs_para = re.search(r"Observed\snumber\sof\sparallel-change\ssites\s*=\s*(\d*\.?\d*)", CONVERG2_out, re.S).group(1)
    prob_para = re.findall(r"probability\s*=\s*(\d*\.?\d*)", CONVERG2_out, re.S)[0]
    exp_conv = re.search(r"Expected\snumber\sof\sconvergent-change\ssites\s*=\s*(\d*\.?\d*)", CONVERG2_out, re.S).group(1)
    obs_conv = re.search(r"Observed\snumber\sof\sconvergent-change\ssites\s*=\s*(\d*\.?\d*)", CONVERG2_out, re.S).group(1)
    prob_conv = re.findall(r"probability\s*=\s*(\d*\.?\d*)", CONVERG2_out, re.S)[1]

print(exp_para,obs_para,prob_para,exp_conv,obs_conv,prob_conv)
"""
import re
import glob
for file in glob.glob("./*rst"):
    with open(file) as f:
        records = f.read()
        tree_branch= re.search(r"Ancestral\sreconstruction\sby\sAAML\.\n+(.+);\n.+;.+\n+tree\swith\snode\slabels\sfor\sRod\sPage",records,re.S).group(1).strip()
        tree_topology_anc = re.search(r"tree\swith\snode\slabels\sfor\sRod\sPage\'s\sTreeView\n(.+);",records,re.S).group(1).strip()
        print(tree_branch)
        print(tree_topology_anc)
        anc_nodes = re.findall(r"\)\s(\d+)",tree_topology_anc)
        print(anc_nodes)
        for anc in anc_nodes:
            temp_tree = re.sub("\):",")"+anc+":",tree_branch,1)
            tree_branch = temp_tree
        tree_branch = tree_branch + anc_nodes[-1] + ":0.00000;"
        tree_branch_anc = re.sub(r"\s+","",tree_branch)
        print(tree_branch_anc)
        print(tree_branch)


from io import StringIO
test = open("test.txt","w")
for file in glob.glob("./*rat"):
    with open(file) as f:
        records = f.read()
        match_content = re.search(r"(\s+Site\s+Freq\s+Data\s+Rate.+)\n+lnL\s+=",records,re.S)
        lst = match_content.group(1).strip().replace("\n\n","\n").split('\n')
        lst1 = [line.strip() for line in lst]
        lst2 = [re.sub(r"\s+",",",line) for line in lst1]
        mat = "\n".join(lst2).replace("posterior,mean,&,category","posterior_mean&category")
        df = pd.read_csv(StringIO(mat), sep=",",header=0)
        print(df["Rate"])
        df["Rate"].to_csv("./test/test.txt",index=False)
import os
print(os.getcwd())

#os.mkdir("test")
print(os.path.join(os.getcwd(),"test"))
folders = ["01_NodeSeq","02_Trees","03_rate","04_siteFreq"]
def creat_folder(folders_list):
    for x in folders_list:
        folder_path = os.path.join(os.getcwd(),x)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print("Created: "+x)
creat_folder(folders)

import shutil
#shutil.move("test.txt",'test')

def rst2dict(rst_records):
    seq1 = rst_records.group(1).strip().split('\n')
    seq2 = [seq.replace(' #', '').partition(" ") for seq in seq1]
    seq3 = {seq[0]: seq[2].strip().replace(' ', '') for seq in seq2}
    return seq3

AA_list = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V','a','r','n','d','c','e','q','g','h','i','l','k','m','f','p','s','t','w','y','v']
from collections import Counter
from pandas import DataFrame
for file in glob.glob("./*rst"):
    with open(file) as f:
        records = f.read()
        all_seqs = re.search(r"List\sof\sextant\sand\sreconstructed\ssequences\n+\s*\d+\s*\d+\n+(.+)\n+Overall\saccuracy\sof", records,re.S)
        if all_seqs is not None:
            seqs_dict = rst2dict(all_seqs)
#        print(seqs_dict)
    siteFreq_dict = {}
    taxon = list(seqs_dict.keys())
    seq_len = len(list(seqs_dict.values())[0])
    seq_num = len(taxon)
    for i in range(0, seq_len):
        each_site_aa_list = []
        for sp in sorted(taxon):
            each_site_aa_list.append(seqs_dict[sp][i])
        each_site_count = Counter(each_site_aa_list)
        aaFreq_each_site = {}
        for aa in AA_list[:20]:
            if aa in each_site_count:
                aaFreq_each_site[aa] = each_site_count[aa]/seq_num
            else:
                aaFreq_each_site[aa] = 0.0
        siteFreq_dict[i] = aaFreq_each_site
        df = DataFrame(siteFreq_dict).T

        df.to_csv("df.csv",index=False,header=False,sep="\t",float_format='%.16f')
        print(df)










