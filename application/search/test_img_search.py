import sys
sys.path.append('./')

from LibVQ.learnable_index import LearnableIndex
from transformers import CLIPProcessor, CLIPModel

from transformers import HfArgumentParser
from search.img_search import img_search
from arguments import IndexArguments, DataArguments, SearchArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, SearchArguments))
    index_args, data_args, search_args = parser.parse_args_into_dataclasses()
    # get index and train
    index = LearnableIndex.load_all(index_args.load_path)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    answer, answer_id = img_search(search_args.query,
                                   index, data_args.collection_path,
                                   model, processor)
    print(answer)
    print(answer_id)