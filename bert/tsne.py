import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

checkpoint_dir='' # folder where checkpoints and qn/passage tokens were saved
embs_dir='' # folder where embeddings are stored

# TSNE
plt.rcParams.update({'font.size': 15})
ds=[]
for i in range(12) : 
	d=np.load(embs_dir+'/emb_enclayer_'+str(i)+'.npy',allow_pickle=True,encoding='latin1').item()
	ds.append(d)
ids=list(ds[0].keys())
tokens=np.load(checkpoint_dir+'/qn_and_doc_tokens.npy',encoding='latin1',allow_pickle=True).item()

#colors=['black','red','blue','green','magenta','cyan','yellow','gray','indigo','darkolivegreen','crimson','slateblue']
qns=range(30,40)
# plotting only questions 30-39 for now

for i in qns : 
	if i%1==0 : 
		print(i)
	words=tokens[ids[i]]
	print(' '.join(words))
	print('\n')
	seq_len=len(words)
	for j in range(12) : 

		start_ind=ds[j][ids[i]]['start_ind']
		end_ind=ds[j][ids[i]]['end_ind']
		words=ds[j][ids[i]]['words']
		#print(start_ind,end_ind,' '.join(words))
		sep1=words.index('[SEP]')
		try : 
			full1=words[(end_ind+1):].index('.')+end_ind
		except : 
			full1=end_ind+5
		try : 
			tmp=words[:start_ind]
			full2=len(tmp)-1-tmp[::-1].index('.')
		except : 
			full2=start_ind-4
		#print(full2,full1,sep1)
		#os.sys.exit()
		tsne_rep=TSNE(n_components=2,perplexity=40,random_state=0).fit_transform(ds[j][ids[i]]['emb'])
		assert tsne_rep.shape[0]==seq_len
		tsne_x=tsne_rep[:,0]
		tsne_y=tsne_rep[:,1]

		plt.figure(figsize=(8,8))
		plt.title('t-SNE plot for Question '+str(qns[i])+', for Layer '+str(j),fontsize=22)

		ll=[0,0,0,0]
		lines=[]	
		labels=[]	
		for k in range(len(tsne_x)) :
			fontcolor='k'
			fontsize=16
			label=''
			if k in range(start_ind,end_ind+1) : # ans span
				color='r'
				ms=600
				marker='o'
				if ll[0]==0 : 
					ll[0]=k	
				label='answer span'
			elif words[k]=='[CLS]' or words[k]=='[SEP]' : 
				color='k'
				ms=250 
				marker='s'
				if ll[1]==0 : 
					ll[1]=k
				label='[CLS]/[SEP]'
			elif k in range(1,sep1) : # qn words
				color='green'
				ms=250
				marker='v'
				if ll[2]==0 : 
					ll[2]=k
				label='query words'
			#elif (k not in range(1,sep1)) and (words[k] in words[1:sep1]) : 
			#	color='slategray'
			#	ms=200
			elif (k in range(full2,start_ind)) or (k in range(end_ind+1,full1)) : # contextual
				color='fuchsia'
				ms=250
				marker='X'
				if ll[3]==0 : 
					ll[3]=k
				label='contextual words'
			else :  
				color='lightsteelblue'
				ms=50
				fontcolor='lightgray'
				fontsize=6
				marker='.' 
			sc=plt.scatter(tsne_x[k],tsne_y[k],c=color,cmap=plt.cm.get_cmap("jet",10),s=ms,marker=marker,label=label)
			#pp,qq=sc.legend_elements()
			lines.append(sc)#pp)
			labels.append(label)#qq)
			plt.annotate(words[k],xy=(tsne_x[k],tsne_y[k]),xytext=(5,2),
				textcoords='offset points',ha='right',va='bottom',fontsize=fontsize,color=fontcolor)
		#plt.colorbar(ticks=range(10))
		final_lines=[]
		final_labels=[]#'answer span','[CLS]/[SEP]','query words','contextual words']
		for k in range(4) :
			final_lines.append(lines[ll[k]])
			final_labels.append(labels[ll[k]])
			#final_labels.append(final_lines[k].get_label())
		print(final_lines)
		print(final_labels)
		plt.legend(handles=final_lines,labels=final_labels,fontsize=16,loc='best')			

		plt.savefig('tsne_plots/tsne_perplexity_40/tsne_qn_'+str(qns[i])+'_layer_'+str(j))
		plt.tight_layout()
		plt.close()
print('TSNE DONE!')


