from LibVQ.dataset import Datasets

from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.base_index import FaissIndex

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

    index = FaissIndex(index_config, encoder_config)
    index.train(data)

    index.save_all(index_args.save_path)
