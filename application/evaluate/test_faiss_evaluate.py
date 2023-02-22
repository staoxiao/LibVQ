import sys
sys.path.append('./')

from LibVQ.dataset import Datasets
from LibVQ.base_index import FaissIndex
from LibVQ.dataset.dataset import load_rel

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    # get index and train
    data = Datasets(data_args.data_dir,
                    emb_size=data_args.data_emb_size,
                    new_query=False)

    index = FaissIndex.load_all(index_args.save_path)

    # MRR and RECALL
    if data.dev_queries_embedding_dir is not None:
        dev_query = index.get_emb(data, data.dev_queries_embedding_dir)
        ground_truths = load_rel(data.dev_rels_path)
        index.test(dev_query, ground_truths, topk=1000, batch_size=64,
                   MRR_cutoffs=[5, 10, 20, 50, 100], Recall_cutoffs=[5, 10, 20, 50, 100],
                   nprobe=index.index_config.nprobe)