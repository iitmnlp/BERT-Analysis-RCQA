'''
-------------------------
JENSON SHANNON DIVERGENCE
-------------------------
'''

from dit.divergences import jensen_shannon_divergence
import numpy as np
import dit
from scipy.spatial import distance

num_examples=1000 # number of examples to be considered for averaging
num_to_remove=2 # number of scores to be removed from the top
num_to_keep=2 # number of scores to be retained at the top

num=12 # number of layers
ig_dir_name='ig_scores' # path to directory where IG scores npy's are stored

# loading all layers' IG scores
d=[]
for i in range(num) : 
	ds=np.load(ig_dir_name+'/importance_scores_ig_'+str(i)+'.npy',encoding='latin1',allow_pickle=True).item()
	d.append(ds)

ids=list(d[0]['imp_scores'].keys())
seq_len=d[0]['seq_len']

# removing top-k scores
qq=[]
for i in range(len(ids[:num_examples])) : 
	if i%100==0 : 
		print(i)
	ct_id=ids[i]
	sl=seq_len[ct_id]

	sep=d[0]['words'][ct_id].index('[SEP]')	
	mat=np.zeros((num,num))
	for j in range(num) : 
		attn1=np.copy(d[j]['imp_scores'][ct_id][sep:sl])

		sorted_inds=np.argsort(attn1)[::-1]
		attn1[sorted_inds[:num_to_remove]]=0

		attn_sum1=np.expand_dims(np.sum(attn1),-1)
		attn1=attn1/attn_sum1
		attn1_=dit.Distribution.from_ndarray(attn1)

		for k in range(num) : 
			attn2=np.copy(d[k]['imp_scores'][ct_id][sep:sl])

			sorted_inds=np.argsort(attn2)[::-1]
			attn2[sorted_inds[:num_to_remove]]=0

			attn_sum2=np.expand_dims(np.sum(attn2),-1)
			attn2=attn2/attn_sum2
			attn2_=dit.Distribution.from_ndarray(attn2)
			
			d1=jensen_shannon_divergence([attn1_,attn2_])

			mat[j][k]=d1
	qq.append(mat)

print('JSD rem '+str(num_to_remove))
np.save(ig_dir_name+'/jenson_shannon_matrix_rem'+num_to_remove+'.npy',{'qq':qq,'ids':ids[:num_examples]})	

# retaining only top-k scores
qq=[]
for i in range(len(ids[:num_examples])) : 
	if i%100==0 : 
		print(i)
	ct_id=ids[i]
	sl=seq_len[ct_id]

	sep=d[0]['words'][ct_id].index('[SEP]')	
	mat=np.zeros((num,num))
	for j in range(num) : 
		attn1=np.copy(d[j]['imp_scores'][ct_id][sep:sl])

		sorted_inds=np.argsort(attn1)[::-1]
		attn1[sorted_inds[num_to_keep:]]=0

		attn_sum1=np.expand_dims(np.sum(attn1),-1)
		attn1=attn1/attn_sum1
		attn1_=dit.Distribution.from_ndarray(attn1)

		for k in range(num) : 
			attn2=np.copy(d[k]['imp_scores'][ct_id][sep:sl])

			sorted_inds=np.argsort(attn2)[::-1]
			attn2[sorted_inds[num_to_keep:]]=0

			attn_sum2=np.expand_dims(np.sum(attn2),-1)
			attn2=attn2/attn_sum2
			attn2_=dit.Distribution.from_ndarray(attn2)
			
			d1=jensen_shannon_divergence([attn1_,attn2_])

			mat[j][k]=d1
	qq.append(mat)

print('JSD keep '+str(num_to_remove))
np.save(ig_dir_name+'/jenson_shannon_matrix_keep'+num_to_keep+'.npy',{'qq':qq,'ids':ids[:num_examples]})	


# JSD as such, without removing any scores
qq=[]
for i in range(len(ids[:num_examples])) : 
	if i%100==0 : 
		print(i)
	ct_id=ids[i]
	sl=seq_len[ct_id]

	sep=d[0]['words'][ct_id].index('[SEP]')	
	mat=np.zeros((num,num))
	for j in range(num) : 
		attn1=d[j]['imp_scores'][ct_id][sep:sl]
		attn_sum1=np.expand_dims(np.sum(attn1),-1)
		attn1=attn1/attn_sum1
		attn1_=dit.Distribution.from_ndarray(attn1)
		for k in range(num) : 
			attn2=d[k]['imp_scores'][ct_id][sep:sl]
			attn_sum2=np.expand_dims(np.sum(attn2),-1)
			attn2=attn2/attn_sum2
			attn2_=dit.Distribution.from_ndarray(attn2)
			
			d1=jensen_shannon_divergence([attn1_,attn2_])

			mat[j][k]=d1
	qq.append(mat)
print('JSD normal',)
np.save(ig_dir_name+'/jenson_shannon_matrix.npy',qq)














