'''
-------------------------------
HEATMAPS ETC FOR JENSON SHANNON
-------------------------------
'''

import numpy as np
import os
import matplotlib.pyplot as plt

num=12 # number of layers
ig_dir_name='ig_scores' # path to directory where IG scores npy's are stored
dataset_name='squad' # name of dataset, for storage purposes - using 'squad' or 'duorc'
num_to_remove=2 # number of scores to be removed from the top
num_to_keep=2 # number of scores to be retained at the top

# normal JSD graph
q1=np.load(ig_dir_name+'/jenson_shannon_matrix.npy')
print(len(q1))
print(q1[0].shape)
qq=[np.average(q1,axis=0)]

for i in range(len(qq)) : 
	mat=qq[i]
	if not os.path.exists('js_heatmaps') : 
		os.makedirs('js_heatmaps')
	path_to_save=os.path.join('js_heatmaps',dataset_name+'_normal_avg_1000')#,str(i))

	plt.figure(figsize=(6,6))
	plt.rc('xtick', labelsize=12) 
	plt.rc('ytick', labelsize=12) 
	font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 12}
	plt.rc('font', **font)
	plt.rcParams["axes.grid"] = False
	ticks = [str(j) for j in range(num)]#sent + ['<eos>']
	#sns.heatmap(attns, annot=True, fmt="f", annot_kws={"size": 8}, center=0.5)
	#extent = (0, mat.shape[1], mat.shape[0], 0)
	plt.imshow(mat, interpolation='none', cmap='Blues',vmin=0,vmax=1)#,extent=extent)
	plt.xticks(range(len(ticks)), ticks, rotation=90,fontsize=12);
	plt.yticks(range(len(ticks)), ticks,fontsize=12);
	#plt.title('BERT - SQuAD Integrated Gradients JSD',fontsize=18)
	plt.title('BERT - '+dataset_name+' Integrated Gradients JSD',fontsize=18)
	for j in range(num) : 
		for k in range(num) : 
			text=plt.text(k,j,round(mat[j,k],2),ha="center",va="center",color="black",fontsize=11.5)
	#plt.colorbar(label='attention weights')
	#plt.grid(which='both',color='k',linestyle='-',linewidth='1')
	plt.tight_layout()
	plt.savefig(path_to_save)
	plt.close()

q1=np.load(ig_dir_name+'/jenson_shannon_matrix_rem'+str(num_to_remove)+'.npy')
print(len(q1))
print(q1[0].shape)
qq=[np.average(q1,axis=0)]
for i in range(len(qq)) : 
	mat=qq[i]
	path_to_save=os.path.join('js_heatmaps',dataset_name+'_rem_'+str(num_to_remove)+'_avg_1000')#,str(i))

	plt.figure(figsize=(6,6))
	plt.rc('xtick', labelsize=12) 
	plt.rc('ytick', labelsize=12) 
	font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 12}
	plt.rc('font', **font)
	plt.rcParams["axes.grid"] = False
	ticks = [str(j) for j in range(num)]#sent + ['<eos>']
	#sns.heatmap(attns, annot=True, fmt="f", annot_kws={"size": 8}, center=0.5)
	#extent = (0, mat.shape[1], mat.shape[0], 0)
	plt.imshow(mat, interpolation='none', cmap='Blues',vmin=0,vmax=1)#,extent=extent)
	plt.xticks(range(len(ticks)), ticks, rotation=90,fontsize=12);
	plt.yticks(range(len(ticks)), ticks,fontsize=12);
	#plt.title('BERT - SQuAD Integrated Gradients JSD \n Top 5 Retained',fontsize=18)
	plt.title('BERT - '+dataset_name+' Integrated Gradients JSD \n Top '+str(num_to_remove)+' Removed',fontsize=18)
	#plt.title('BERT - DuoRC Integrated Gradients JSD',fontsize=18)
	for j in range(num) : 
		for k in range(num) : 
			text=plt.text(k,j,round(mat[j,k],2),ha="center",va="center",color="black",fontsize=11.5)
	#plt.colorbar(label='attention weights')
	#plt.grid(which='both',color='k',linestyle='-',linewidth='1')
	plt.tight_layout()
	plt.savefig(path_to_save)
	plt.close()


q1=np.load(ig_dir_name+'/jenson_shannon_matrix_keep'+str(num_to_keep)+'.npy')
print(len(q1))
print(q1[0].shape)
qq=[np.average(q1,axis=0)]
for i in range(len(qq)) : 
	mat=qq[i]
	path_to_save=os.path.join('js_heatmaps',dataset_name+'_rem_'+str(num_to_keep)+'_avg_1000')#,str(i))

	plt.figure(figsize=(6,6))
	plt.rc('xtick', labelsize=12) 
	plt.rc('ytick', labelsize=12) 
	font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 12}
	plt.rc('font', **font)
	plt.rcParams["axes.grid"] = False
	ticks = [str(j) for j in range(num)]#sent + ['<eos>']
	#sns.heatmap(attns, annot=True, fmt="f", annot_kws={"size": 8}, center=0.5)
	#extent = (0, mat.shape[1], mat.shape[0], 0)
	plt.imshow(mat, interpolation='none', cmap='Blues',vmin=0,vmax=1)#,extent=extent)
	plt.xticks(range(len(ticks)), ticks, rotation=90,fontsize=12);
	plt.yticks(range(len(ticks)), ticks,fontsize=12);
	#plt.title('BERT - SQuAD Integrated Gradients JSD \n Top 5 Retained',fontsize=18)
	plt.title('BERT - '+dataset_name+' Integrated Gradients JSD \n Top '+str(num_to_keep)+' Removed',fontsize=18)
	#plt.title('BERT - DuoRC Integrated Gradients JSD',fontsize=18)
	for j in range(num) : 
		for k in range(num) : 
			text=plt.text(k,j,round(mat[j,k],2),ha="center",va="center",color="black",fontsize=11.5)
	#plt.colorbar(label='attention weights')
	#plt.grid(which='both',color='k',linestyle='-',linewidth='1')
	plt.tight_layout()
	plt.savefig(path_to_save)
	plt.close()




















