from dataclasses import dataclass, field

@dataclass
class IndexArguments:
    index_method: str = field(default='ivfopq', metadata={"help": "The faiss index type, should be in [ivfopq, ivfpq, ivf, opq, pq]"})
    dist_mode: str = field(default='ip', metadata={"help": "How calculate the distance between embeddings. ip or l2"})
    ivf_centers_num: int = field(default=10000, metadata={"help": "The number of post list in ivf index"})
    subvector_num: int = field(default=64, metadata={"help": "The number of subvector"})
    subvector_bits: int = field(default=8, metadata={"help": "The bits each subvector will be quantized to; k bits means 2^k codeword in each subspace"})
    nprobe: int = field(default=100, metadata={"help": "The number of post list which need to be searched"})


@dataclass
class DataArguments:
    data_dir: str = field(default='./data/passage/dataset', metadata={"help": "The path to preprocessed data"})
    preprocess_dir: str = field(default='./data/passage/preprocess', metadata={"help": "The path to preprocessed data"})
    neg_file: str = field(default=None, metadata={"help": "The path to negative.json"})
    doc_embeddings_file: str = field(default=None, metadata={"help": "The path to embeddings"})
    query_embeddings_file: str = field(default=None, metadata={"help": "The path to embeddings"})
    save_ckpt_dir: str = field(default='./saved_ckpts/')
    load_encoder_dir: str = field(default=None)
    output_dir: str = field(default=None)
    savename: str = field(default='LearnableIndex')
    max_query_length: int = field(default=24)
    max_doc_length: int = field(default=512)


@dataclass
class ModelArguments:
    pretrained_model_name: str = field(default='bert-base-uncased')
    use_two_encoder: bool = field(default=True)
    sentence_pooling_method: str = field(default='first')


@dataclass
class TrainingArguments:
    master_port: str = field(default='13455')

    training_mode: str = field(default='distill')

    optimizer_str: str = field(default="adaw")
    loss_method: str = field(default='distill')
    temperature: float=field(default=1)
    max_grad_norm: float = field(default=1)

    per_query_neg_num: int = field(default=1)
    fix_emb: str = field(default='doc', metadata={"help": "Fix the embeddings of doc and load them from infer_path"})
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}, )

    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for Adam."})
    num_train_epochs: int = field(default=5.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
