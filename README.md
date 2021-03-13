# BERT-Analysis-RCQA
**Official code for paper "Towards Interpreting BERT for Reading Comprehension Based QA"**
*Code upload in progress.*

*This code was developed on top of the [official BERT code released by Google](https://github.com/google-research/bert).*

Download the required base model checkpoints from the official git repository.
We used `uncased_L-12_H-768_A-12`, which has the following configuration: ``12-layer, 768-hidden, 12-heads, 110M parameters``

To finetune the BERT model on SQuAD/DuoRC - 
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


