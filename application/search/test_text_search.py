import sys
sys.path.append('./')

from LibVQ.learnable_index import LearnableIndex

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments, SearchArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, SearchArguments))
    index_args, data_args, search_args = parser.parse_args_into_dataclasses()
    # get index and train
    index = LearnableIndex.load_all(index_args.load_path)
    # search
    answer, answer_id = index.search_query(search_args.query, data_args.collection_path)
    print(answer)
    print(answer_id)