import os
import pickle
import re
import shutil
import sys
import json
import argparse
import random
import csv
import copy
import collections
import string
#from Transparency.preprocess import vectorizer

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

from collections import defaultdict

np.set_printoptions(suppress=True)

import nltk
nltk.data.path.append('..')
nltk.data.path.append('../corpora')
entity_pos_tags=['CD','NN','NNP','NNS','PRP','PRP$']
adjective_pos_tags=['JJ','JJR','JJS']
verb_pos_tags=['VB','VBD','VBG','VBN','VBP','VBZ']
stopword_pos_tags=['CC','DT','EX','IN','LS','MD','PDT','POS','RP','TO','UH','WDT','WP','WRB']
adverb_pos_tags=['RB','RBR','RBS']
punct_marks=list(string.punctuation)

checkpoint_dir='' # folder where checkpoints and qn/passage tokens were saved
ig_dir='' # folder where embeddings are stored

parser=argparse.ArgumentParser()
parser.add_argument('--dummy',type=int,default=0,help='')
args=parser.parse_args()

names=[]
for i in range(12) : 
	names.append('encoding_layer_'+str(i))

ds=[]
for i,name in enumerate(names) :
	if i in [] :
		ds.append([])
		continue
	d=np.load(ig_dir+'/importance_scores_ig_'+str(i)+'.npy',allow_pickle=True,encoding='latin1').item()
	ds.append(d)

nbest_json_file=open(checkpoint_dir+'/nbest_predictions.json')
nbest_json=json.load(nbest_json_file)
probab_quantifier=[]
probab_not_quantifier=[]
quant_check1=0
quant_check2=0
quant_ig_diffs=[]

exact_scores=np.load(checkpoint_dir+'/exact_scores.npy').item()
predictions=json.load(open(checkpoint_dir+'/predictions.json'))
tokens=np.load(checkpoint_dir+'/qn_and_doc_tokens.npy',encoding='latin1').item()

list_=range(1000)
ids=list(d['words'].keys())
layers_num=collections.defaultdict(list)
quantifier_correct_num=0
quantifier_correct_den=0

for i in range(len(ids)) : #list_ :
	if i%1000==0 : 
		print(i)
	id_ct=ids[i] 

	s=tokens[id_ct]
	seq_len=len(s)

	temp_sep_token=ds[0]['words'][id_ct].index('[SEP]')
	q=' '.join(ds[0]['words'][id_ct][1:temp_sep_token])

	pred_ans=predictions[id_ct]

	# cleaning data
	# finding ##s
	cnters=[]
	cnter=0
	j=0

	while j<seq_len : 
		tmp=[]
		#if s[j]=="[SEP]" and sep1!=0 : 
		#	sep1=j
		while "##" in s[j] : 
			tmp.append(j)
			j=j+1 
		if len(tmp)!=0 : 
			j=j-1
			cnters.append(tmp)
		j=j+1
	#print("Finding ##s done")

	# combing words with ##s
	s_new=[]
	prev=0
	for j in range(len(cnters)) : 
		tmp=cnters[j]
		combined="".join(s[tmp[0]:(tmp[-1]+1)])
		combined=combined.replace(" ##","")
		combined=combined.replace("##","")
		s_new = s_new + s[prev:tmp[0]] + [combined]
		prev=tmp[-1]+1

	# adding remaining words
	s_new = s_new + s[prev:]
	s=s_new 
	sep1=s.index("[SEP]")+1	
	qwords=copy.deepcopy(s[1:sep1])
	s=s[sep1:-1]
	pos=nltk.pos_tag(s)
	#print("Combining ##s done")

	# To get percentage of quantifier questions that are predicted correctly
	if 'how' in qwords and ('many' in qwords or 'much' in qwords) : 
		quantifier_correct_num+=exact_scores[id_ct]
		quantifier_correct_den+=1

	# To get confidence of model on quantifier questions
	prob=nbest_json[id_ct][0]["probability"]
	if 'how' in qwords and ('many' in qwords or 'much' in qwords) : 
		probab_quantifier.append(prob)
	else : 	
		probab_not_quantifier.append(prob)	

	for j,name in enumerate(names) : 
		flag_qwords=0	

		attn=ds[j]['imp_scores'][id_ct].reshape(384)[:seq_len]
		
		# adding up scores of words which have ##s
		attn_new=[]
		prev=0
		for k in range(len(cnters)) : 
			tmp=cnters[k]
			comb=np.sum(attn[tmp[0]:(tmp[-1]+1)])
			attn_new = attn_new + list(attn[prev:tmp[0]]) + [comb]
			prev=tmp[-1]+1
		attn_new = attn_new + list(attn[prev:])
		attn=np.array(attn_new)
		#assert len(attn)==len(s)

		attn=attn[sep1:-1]
		assert len(attn)==len(s)
		attn_sum=np.expand_dims(np.sum(attn),-1)
		attn=attn/attn_sum		

		# getting just top 5 tokens from attn distribution
		argsorted=np.argsort(attn)[::-1]
		other_than_top5=argsorted[5:]
		attn[other_than_top5]=0		

		w=''
		wl=[]
		wll=[]
		for ii in argsorted[:5] : 
			w=w+' '+s[ii]
			wl.append(s[ii])
			wll.append(ii)

		flag1=0
		flag2=0
		flag3=0
		f1=[]
		f2=[]	
		
		# QUANTIFIER QUESTIONS
		if 'how' in qwords and ('many' in qwords or 'much' in qwords) : 
			#with open('quantifier_question_details.txt','a') as f : 
			#	f.write('Question : '+str(i)+', Layer : '+str(j)+', Answer : '+predictions[id_ct]+'\n')
			for ii in range(len(s)) : 
				if pos[ii][1]=='CD' : flag1+=1
			tmp_quant_words=[]
			for ii in argsorted[:5] : 
				if pos[ii][1]=='CD' : 
					if s[ii] in predictions[id_ct] : flag3=1
					flag2+=1
					tmp_quant_words.append((s[ii],attn[ii]))
					# Writing IG scores into a text file
					#with open('quantifier_question_details.txt','a') as f :
					#	f.write(s[ii]+' : '+str(attn[ii])+',, ')

			# if len(tmp_quant_words)>1 and j==10 : 
			# 	if (tmp_quant_words[0][0] in predictions[id_ct]) and (tmp_quant_words[1][0] not in predictions[id_ct]) : 
			# 		quant_ig_diffs.append(tmp_quant_words[0][1]-tmp_quant_words[1][1])
						 
			if flag3==1 and flag2>0 : flag2=flag2-1
				
		
		if flag1>1 : 
			layers_num[j].append(float(flag2)/flag1)
		# if j==10 and flag1>1 : 
		# 	tmp=float(flag2)/flag1
		# 	if tmp>0.2 :
		# 		quant_check2+=1
		# 		quant_check1+=exact_scores[id_ct]

	
print(layers_num.keys())
for i in range(12) : 
	print('Layer ', i,' : ',round(np.mean(layers_num[i]),4))
# for i in range(12) : 
# 	print(round(np.mean(layers_num[i]),2))

print('\nTotal number of quantifier questions : ',quantifier_correct_den)
print('Number of quantifier questions BERT answered correctly : ',quantifier_correct_num)
print('% of quantifier questions BERT answered correctly : ',float(quantifier_correct_num)/quantifier_correct_den,'\n')

print('\nBERT\'s Confidence in the answer')
print('For quantifier questions : ')
print('Mean : ',np.mean(probab_quantifier),'Var : ',np.var(probab_quantifier),' Median : ',np.median(probab_quantifier))
print('For non-quantifier questions : ')
print('Mean : ',np.mean(probab_not_quantifier),'Var : ',np.var(probab_not_quantifier),' Median : ',np.median(probab_not_quantifier))
# print('\nChecking how many questions that have more than 2 num entities in the top-5 words are correct')
# print('Quant check : ',quant_check1,quant_check2)
# print('\nChecking the difference b/w ans-span and not-ans-span num entity words\' ig scores')
# print('Len : ',len(quant_ig_diffs),' Mean : ',np.mean(quant_ig_diffs),' Var : ',np.var(quant_ig_diffs),' Median : ',np.median(quant_ig_diffs))
