import sys
sys.path.append('./')

from topic_modeling.topic_model import TopicModel, get_classes

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments))
    index_args, data_args = parser.parse_args_into_dataclasses()
    # get index and train

    classes = get_classes(index_args.collection_path, index_args.index_path)

    topic_model = TopicModel(classes)
    topic_model.get_all_topic_info()
    topic_model.get_topic(2)