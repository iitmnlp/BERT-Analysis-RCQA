# BERT-Analysis-RCQA
**Official code for paper [Towards Interpreting BERT for Reading Comprehension Based QA](https://www.aclweb.org/anthology/2020.emnlp-main.261).**

*This code was developed on top of the [official BERT code released by Google](https://github.com/google-research/bert).*

## Experimental Setup
The base model checkpoints for BERT can be downloaded from the [official git repository](https://github.com/google-research/bert). We used `uncased_L-12_H-768_A-12`, which has the following configuration: ``12-layer, 768-hidden, 12-heads, 110M parameters``. All codes have to be run inside the ```bert``` directory.

We used the SQuAD-V1 and Self-RC (DuoRC) datasets - all data processing is taken care of by the code itself, before training or evaluation.

We used `tensorflow-v1.14` with Python 2 for training/evaluation, and Python 3 for all analyses.

## To finetune the BERT model on SQuAD/DuoRC:
Use ```run_squad_infer.py``` for SQuAD and ```duorc_infer.py``` for DuoRC.
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

## To evaluate the BERT model's predictions:
Use ```evaluate-v2.0.py``` for SQuAD and ```duorc_evaluate-v2.0.py``` for DuoRC.
```
python -u evaluate-v2.0.py /path/to/dataset/json /path/to/predictions/json --checkpoint_dir=/path/to/checkpoints/folder
```

## To generate integrated gradient scores for each layer:
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

## Codes to generate Jensen-Shannon graphs and save them: 
Use ```jensen_shannon.py``` and ```graph_js.py```.

<img src="https://user-images.githubusercontent.com/17588365/111081891-75b7cd80-852b-11eb-800b-8deeba82bbfc.png" width="300"> <img src="https://user-images.githubusercontent.com/17588365/111082082-638a5f00-852c-11eb-911d-c8ebc8bfa003.png" width="300">

<img src="https://user-images.githubusercontent.com/17588365/111082094-7ac94c80-852c-11eb-9844-77cd1c483f14.png" width="300"> <img src="https://user-images.githubusercontent.com/17588365/111082103-84eb4b00-852c-11eb-8340-f05c2bd57437.png" width="300">


## Code to generate t-SNE plots: 
Use ```tsne.py``` (this code uses the embeddings saved by the IG scores code above).

<img src="https://user-images.githubusercontent.com/17588365/111082184-f5926780-852c-11eb-9768-459b8fea9bed.png" width="300"> <img src="https://user-images.githubusercontent.com/17588365/111082186-f9be8500-852c-11eb-80e2-c5fb4c45e38f.png" width="300">

<img src="https://user-images.githubusercontent.com/17588365/111082193-fc20df00-852c-11eb-8331-ccf616638613.png" width="300"> <img src="https://user-images.githubusercontent.com/17588365/111082196-ff1bcf80-852c-11eb-97b4-c463122f0387.png" width="300">


## Code to analyze quantifier questions:
Use ```quantifier_questions_analysis.py```.

## Contact
For any questions, please contact sahanjich@gmail.com.