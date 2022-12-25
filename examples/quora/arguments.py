from dataclasses import dataclass, field
from typing import List

@dataclass
class IndexArguments:
    index_method: str = field(default='opq',
                              metadata={"help": "The faiss index type, should be in [ivf_opq, ivf_pq, ivf, opq, pq]"})
    dist_mode: str = field(default='ip', metadata={"help": "How calculate the distance between embeddings. ip or l2"})
    ivf_centers_num: int = field(default=None, metadata={"help": "The number of post list in ivf index"})
    subvector_num: int = field(default=64, metadata={"help": "The number of subvector"})
    subvector_bits: int = field(default=8, metadata={
        "help": "The bits each subvector will be quantized to; k bits means 2^k codeword in each subspace"})
    nprobe: int = field(default=1, metadata={"help": "The number of post list which need to be searched"})
    emb_size: int = field(default=768, metadata={"help": "The dimension of embedding"})
    save_path: str = field(default='./data/quora/parameters', metadata={"help": "Path of all parameters of save index"})
    load_path: str = field(default=None, metadata={"help": "Path of all parameters of load index"})
    index_path: str = field(default=None, metadata={"help": "Path of index"})

@dataclass
class DataArguments:
    data_dir: str = field(default='./data/quora', metadata={"help": "The path to all data"})
    collection_path: str = field(default=None, metadata={"help": "The path to collection.tsv"})
    neg_file: str = field(default=None, metadata={"help": "The path to negative.json"})
    save_ckpt_dir: str = field(default='./saved_ckpts/')
    load_encoder_dir: str = field(default=None)
    output_dir: str = field(default=None)
    max_query_length: int = field(default=32)
    max_doc_length: int = field(default=256)
    data_emb_size: int = field(default=None)
    sample_emb_path: str = field(default=None, metadata={"help": "The sample embedding path"})
    list_path: str = field(default='classes.json', metadata={"help": "The path to save and load topic classes"})

@dataclass
class ModelArguments:
    is_finetune: bool = field(default=True)
    doc_encoder_name_or_path: str = field(default='Shitao/RetroMAE_MSMARCO_distill')
    query_encoder_name_or_path: str = field(default='Shitao/RetroMAE_MSMARCO_distill')

@dataclass
class TrainingArguments:
    master_port: str = field(default='13455')

    temperature: float = field(default=1)
    max_grad_norm: float = field(default=-1)

    per_query_neg_num: int = field(default=1)
    fix_emb: str = field(default='doc', metadata={"help": "Fix the embeddings of doc(search2)"})
    per_device_train_batch_size: int = field(
        default=512, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})

    num_train_epochs: int = field(default=16, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    checkpoint_save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})

@dataclass
class SearchArguments:
    query: List[str]
    # search2: str = field(default='what is mean streaming')