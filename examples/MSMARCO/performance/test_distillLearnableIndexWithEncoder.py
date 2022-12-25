import sys
sys.path.append('./')

from LibVQ.dataset import Datasets
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.learnable_index import DistillLearnableIndexWithEncoder
from LibVQ.dataset.dataset import load_rel
import os

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    # get index and train
    data = Datasets(data_args.data_dir,
                    emb_size=data_args.data_emb_size)
    index_config = IndexConfig(index_method=index_args.index_method,
                               ivf_centers_num=index_args.ivf_centers_num,
                               subvector_num=index_args.subvector_num,
                               subvector_bits=index_args.subvector_bits,
                               nprobe=index_args.nprobe,
                               emb_size=index_args.emb_size)
    encoder_config = EncoderConfig(is_finetune=model_args.is_finetune,
                                   doc_encoder_name_or_path=model_args.doc_encoder_name_or_path,
                                   query_encoder_name_or_path=model_args.query_encoder_name_or_path)

    index = DistillLearnableIndexWithEncoder(index_config, encoder_config)
    index.train(data=data,
                temperature=training_args.temperature,
                max_grad_norm=training_args.max_grad_norm,
                per_query_neg_num=training_args.per_query_neg_num,
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                logging_steps=training_args.logging_steps,
                epochs=training_args.num_train_epochs)

    index.save_all(index_args.save_path)

    # MRR and RECALL
    if data.dev_queries_embedding_dir is not None:
        dev_query = index.get(data, data.dev_queries_embedding_dir)
        ground_truths = load_rel(data.dev_rels_path)
        index.test(dev_query, ground_truths, topk=1000, batch_size=64,
                   MRR_cutoffs=[5, 10, 20, 50, 100], Recall_cutoffs=[5, 10, 20, 50, 100], nprobe=index_config.nprobe)

