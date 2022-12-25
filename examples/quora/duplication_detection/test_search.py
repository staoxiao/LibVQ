import sys
sys.path.append('./')

from LibVQ.learnable_index import LearnableIndex
from LibVQ.dataset import Datasets

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments, SearchArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, SearchArguments))
    index_args, data_args, search_args = parser.parse_args_into_dataclasses()
    # get index and train
    data = Datasets(data_args.data_dir,
                    emb_size=data_args.data_emb_size)
    index = LearnableIndex.load_all(index_args.load_path)
    if data.docs_path is not None:
        answer, answer_id = index.search_query(search_args.query, data)
        print(answer)
        print(answer_id)