from dataclasses import dataclass, field

@dataclass
class IndexArguments:
    index_method: str = field(default='ivfopq', metadata={"help": "The faiss index type, should be in [ivfopq, ivfpq, ivf, opq, pq]"})
    ivf_centers_num: int = field(default=10000, metadata={"help": "The number of post list in ivf index"})
    subvector_num: int = field(default=64, metadata={"help": "The number of subvector"})
    subvector_bits: int = field(default=8, metadata={"help": "The bits each subvector will be quantized to; k bits means 2^k codeword in each subspace"})
    nprobe: int = field(default=100, metadata={"help": "The number of post list which need to be searched"})


@dataclass
class DataArguments:
    data_dir: str = field(default='./data/passage/preprocess', metadata={"help": "The path to preprocessed data"})
    neg_file: str = field(default=None, metadata={"help": "The path to negative.json"})
    embeddings_path: str = field(default=None, metadata={"help": "The path to embeddings"})
    save_ckpt_dir: str = field(default=None)
    load_ckpt_dir: str = field(default=None)
    output_dir: str = field(default=None)
    savename: str = field(default='LearnableIndex')
    max_query_length: int = field(default=24)
    max_doc_length: int = field(default=512)


@dataclass
class ModelArguments:
    pretrained_model_name: str = field(default='bert-base-uncased')
    use_two_encoder: bool = field(default=True)


@dataclass
class TrainingArguments:
    master_port: str=field(default='13455')

    optimizer_str: str = field(default="adaw")
    loss_method: str = field(default='multi_ce')
    temperature: float=field(default=1)
    max_grad_norm: float = field(default=1)

    per_query_neg_num: int = field(default=1)
    fix_emb: str = field(default=None, metadata={"help": "Fix the embeddings of doc and load them from infer_path"})
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}, )

    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for Adam."})
    num_train_epochs: float = field(default=100.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
