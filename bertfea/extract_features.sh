export BERT_BASE_DIR=/mnt/raid5/data3/yfliu/B-cell_Epitope/Bert-Protein-master
export BERT_DIR=/mnt/raid5/data3/yfliu/antibacterial_peptide_bert/antibacterial_peptide
python3 /mnt/raid5/data3/yfliu/B-cell_Epitope/Bert-Protein-master/extract_features.py \
  --input_file=$BERT_DIR/datasets/neuropeptides/extra_fea/test_pos.txt \
  --output_file=$BERT_DIR/bertfea/protein/neuropeptides/test/test_pos.json \
  --layers=-1 \
  --vocab_file=$BERT_BASE_DIR/vocab/vocab_1kmer.txt \
  --bert_config_file=$BERT_DIR/pre-bert/json/bert_config_1.json \
  --init_checkpoint=$BERT_BASE_DIR/model/1kmer_model/model.ckpt \
  --max_seq_length=128 \
  --batch_size=32
