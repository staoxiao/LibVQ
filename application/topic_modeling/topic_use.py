import sys
sys.path.append('./')

from topic_modeling.topic_model import TopicModel, get_documents

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments))
    index_args, data_args = parser.parse_args_into_dataclasses()
    # get index and train

    source_documents = get_documents(index_args.collection_path, index_args.index_path)

    topic_model = TopicModel()
    topic_model.fit(source_documents)
    print(topic_model.get_topic_info())
    print(topic_model.get_topic(0))
    print(topic_model.get_document_info())
