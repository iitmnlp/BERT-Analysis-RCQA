# BERT-Analysis-RCQA
**Official code for paper "Towards Interpreting BERT for Reading Comprehension Based QA"**
*(Code upload in progress)*

*This code was developed on top of the [official BERT code released by Google](https://github.com/google-research/bert).*

Download the required base model checkpoints from the official git repository.
We used `uncased_L-12_H-768_A-12`, which has the following configuration: ``12-layer, 768-hidden, 12-heads, 110M parameters``.
All codes have to be run inside the ```bert``` directory.

To finetune the BERT model on SQuAD/DuoRC:
* Use ```run_squad_infer.py``` for SQuAD and ```duorc_infer.py``` for DuoRC.
```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export SQUAD_DIR=/path/to/json-datasets

python -u run_squad_infer.py 
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-json-file \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-json-file \
  --train_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/path/to/new/model/folder
```

To evaluate the BERT model's predictions (```evaluate-v2.0.py``` for SQuAD and ```duorc_evaluate-v2.0.py``` for DuoRC):
```
python -u evaluate-v2.0.py /path/to/dataset/json /path/to/predictions/json --checkpoint_dir=/path/to/checkpoints/folder
```

To generate integrated gradient scores for each layer:
* set layer number from 0 to 11 in ```do_integrated_grad```
* ```predict_batch_size``` must always be set to 1 in this code
* The IG scores are stored in a folder called ```ig_scores``` (for SQuAD) and ```ig_scores_duorc``` (for DuoRC), in files named in the format ```importance_scores_ig_layer_number.npy``` 
* Embeddings for the first 200 datapoints are stored in a folder called ```embs``` (for SQuAD) and ```embs_duorc``` (for DuoRC), in files named in the format ```emb_enclayer_layer_number.npy``` 
* The tokenized 'QN [SEP] PASSAGE' will be stored in ```output_dir/qn_and_doc_tokens.npy```
* Use ```bert_dec_flips.py``` for SQuAD and ```duorc_dec_flips.py``` for DuoRC (since the datasets need to be processed differently).

```
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export SQUAD_DIR=/path/to/json-datasets

python -u bert_dec_flips.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-json-file \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-json-file \
  --train_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --predict_batch_size=1 \
  --do_integrated_grad=11 \
  --output_dir=/path/to/new/model/folder
```

Codes needed to generate Jensen-Shannon graphs and save them: ```jensen_shannon.py``` and ```graph_js.py```.

Code to generate t-SNE plots: ```tsne.py``` (this code uses the embeddings saved by the IG scores code above).