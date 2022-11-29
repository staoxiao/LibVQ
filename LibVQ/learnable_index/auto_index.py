from LibVQ.learnable_index import DistillLearnableIndex, DistillLearnableIndexWithEncoder, ConstrativeLearnableIndex, ConstrativeLearnableIndexWithEncoder
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.dataset import Datasets

class AutoIndex():
    @classmethod
    def get_index(self,
                 index_config: IndexConfig = None,
                 encoder_config: EncoderConfig = None,
                 data: Datasets = None
                 ):
        if index_config is None:
            raise ValueError("You must provide your index config")
        if data is None:
            raise ValueError("You must provide your data")

        if encoder_config is not None and encoder_config.query_encoder_name_or_path is not None and encoder_config.doc_encoder_name_or_path is not None:
            if encoder_config.is_finetune is True:
                if data.docs_path is None:
                    print('your index is distill learnable index')
                    return DistillLearnableIndex(index_config, encoder_config)
                else:
                    print('your index is distill learnable index with encoder')
                    return DistillLearnableIndexWithEncoder(index_config, encoder_config)
            else:
                if data.docs_path is None:
                    print('your index is constrative learnable index')
                    return ConstrativeLearnableIndex(index_config, encoder_config)
                else:
                    print('your index is constrative learnable index with encoder')
                    return ConstrativeLearnableIndexWithEncoder(index_config, encoder_config)

        else:
            print('your index is distill learnable index')
            if data.doc_embeddings_dir is None:
                raise ValueError("Due to the lack of embedding file, you can't use this index")
            return DistillLearnableIndex(index_config, encoder_config)
