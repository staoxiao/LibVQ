# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
    data_dir: str = field()
    hardneg_json: str = field(default=None, metadata={"help": "A Dict store the hard negatives for each query, whose format is {qid:[neg_1, neg_2,...]}."})
    max_query_length: int = field(default=24)
    max_doc_length: int = field(default=512)
    save_model_path: str = field(default='./model')
    savename: str = field(default='BiDR')


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    gradient_checkpointing: bool = field(default=False)
    pq_path: str = field(default=None)
    num_hidden_layers:int=field(default=-1)
    output_embedding_size:int=field(default=-1)

@dataclass
class MyTrainingArguments():
    world_size: int = field(default=4)
    gpu_rank:str=field(default='0,1,2,3')
    master_port:str=field(default='13455')

    optimizer_str: str = field(default="lamb")
    loss_method: str = field(default='multi_ce')
    temperature:float=field(default=1)
    max_grad_norm: float = field(default=1)

    per_query_neg_num: int = field(default=1)
    mink: int=field(default=0)
    maxk: int=field(default=200)
    n2_mink: int=field(default=200)
    n2_maxk: int=field(default=400)
    generate_batch_method: str=field(default='random')
    multi_graph:int=field(default=0)
    patch_num:int=field(default=1)

    fix_doc_emb: bool = field(default=False, metadata={"help": "Fix the embeddings of doc and load them from infer_path"})
    infer_path:str=field(default=None)

    use_linear: bool = field(default=True)
    use_pq: bool = field(default=False)
    partition: int = field(default=96, metadata={"help": "Max gradient norm."})
    centroids: int = field(default=256, metadata={"help": "Max gradient norm."})
    quantloss_weight: float = field(default=1e-6, metadata={"help": "weight for traditional quantization loss (l2 distance)"})
    init_index_path: str = field(default=None, metadata={"help": "use faiss index to init the pq"})

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}, )
    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    num_train_epochs: float = field(default=100.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})
