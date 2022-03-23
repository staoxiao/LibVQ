from dataclasses import dataclass, field


@dataclass
class IndexArguments:
    index_method: str = field(default='ivfopq',
                              metadata={"help": "The faiss index type, should be in [ivfopq, ivfpq, ivf, opq, pq]"})
    dist_mode: str = field(default='ip', metadata={"help": "How calculate the distance between embeddings. ip or l2"})
    ivf_centers_num: int = field(default=10000, metadata={"help": "The number of post list in ivf index"})
    subvector_num: int = field(default=64, metadata={"help": "The number of subvector"})
    subvector_bits: int = field(default=8, metadata={
        "help": "The bits each subvector will be quantized to; k bits means 2^k codeword in each subspace"})
    nprobe: int = field(default=100, metadata={"help": "The number of post list which need to be searched"})


@dataclass
class DataArguments:
    data_dir: str = field(default='./data/passage/dataset', metadata={"help": "The path to preprocessed data"})
    preprocess_dir: str = field(default='./data/passage/preprocess', metadata={"help": "The path to preprocessed data"})
    tokenizer_name: str = field(default='bert-base-uncased', metadata={"help": "The type of tokenizer"})
    neg_file: str = field(default=None, metadata={"help": "The path to negative.json"})
    embeddings_dir: str = field(default=None, metadata={
        "help": "The path to embeddings, including embeddings of docs and queries"})
    save_ckpt_dir: str = field(default='./saved_ckpts/')
    load_encoder_dir: str = field(default=None)
    output_dir: str = field(default=None)
    max_query_length: int = field(default=24)
    max_doc_length: int = field(default=512)


@dataclass
class ModelArguments:
    pass


@dataclass
class TrainingArguments:
    master_port: str = field(default='13455')

    training_mode: str = field(default='distill')

    optimizer_str: str = field(default="adaw")
    loss_method: str = field(default='distill')
    temperature: float = field(default=1)
    max_grad_norm: float = field(default=-1)

    per_query_neg_num: int = field(default=1)
    fix_emb: str = field(default='doc', metadata={"help": "Fix the embeddings of doc(query)"})
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})

    # encoder_lr: float = field(default=1e-5, metadata={"help": "The learning rate for encoder."})
    # ivf_lr: float = field(default=1e-3, metadata={"help": "The learning rate for ivf centers."})
    # pq_lr: float = field(default=1e-4, metadata={"help": "The learning rate for pq codebooks."})

    num_train_epochs: int = field(default=5.0, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    checkpoint_save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
