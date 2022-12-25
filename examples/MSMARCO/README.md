# MSMARCO Dataset using
We take the Passage dataset as the example.   

## Getting Started

### Preparing MSMARCO Data
Download data and convert to our format:
```
bash ./prepare_data/download_data.sh
```
The data will be saved into `./data/MSMARCO`.

### Building index, viewing performance
We build index by the following method:

distill learnable index

```
python ./performance/test_distillLearnableIndex.py \
--data_dir ./data/MSMARCO \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/MSMARCO/parameters \
--per_device_train_batch_size 512
```
if your data_emb_size != index_emb_size:
```
python ./performance/test_distillLearnableIndex.py \
--data_dir ./data/MSMARCO \
--index_method pq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 512 \
--data_emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/MSMARCO/parameters \
--per_device_train_batch_size 512
```
contrastive learnable index

```
python ./performance/test_contrastiveLearnableIndex.py \
--data_dir ./data/MSMARCO \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/MSMARCO/parameters \
--per_device_train_batch_size 512
```

distill learnable index with encoder

```
python ./performance/test_distillLearnableIndexWithEncoder.py \
--data_dir ./data/MSMARCO \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/MSMARCO/parameters \
--per_device_train_batch_size 512
```

contrastive learnable index with encoder

```
python ./performance/test_contrastiveLearnableIndexWithEncoder.py \
--data_dir ./data/MSMARCO \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/MSMARCO/parameters \
--per_device_train_batch_size 512
```

auto index

```
python ./performance/test_autoIndex.py \
--data_dir ./data/MSMARCO \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/MSMARCO/parameters \
--per_device_train_batch_size 512
```

### search
To use the search function here, you must have built the index, and then pass in the save path of the parameters of index and query statement
```
python ./search/test_search.py \
--load_path ./data/MSMARCO/parameters \
--data_emb_size 768 \
--query 'what is paranoid sc' 'what is mean streaming'
```

### embedding compression
To use the search function here, you must have built the pq/opq index(and you can only use pq/opq index), and then pass in the save path of the index and embedding path(memmap path)
```
python ./embedding_compression/test_emb_compression.py \
--index_path ./data/MSMARCO/parameters/learnable.index \
--sample_emb_path  ./data/MSMARCO/embedding/dev-queries.memmap \
--data_emb_size 768
```


### topic modeling
To use the search function here, you must have built the ivf index(ivf/ivf_pq/ivf_opq), and then pass in the save path of the index and collection path
```
pip install scikit-learn
pip install datasets
python ./topic_modeling/topic_use.py \
--collection_path ./data/MSMARCO/collection.tsv \
--index_path  ./data/MSMARCO/parameters/learnable.index \
```

## Quick Started

### Preparing MSMARCO Data

Download data and convert to our format:

```
bash ./prepare_data/download_data.sh
```

The data will be saved into `./data/MSMARCO`.

### Building index, viewing performance

We start by building index by auto_index:

```python
from LibVQ.dataset import Datasets
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.learnable_index import AutoIndex

data = Datasets('./data/MSMARCO')
index_config = IndexConfig(index_method='ivf_pq',
                           ivf_centers_num=10000,
                           subvector_num=64,
                           subvector_bits=8,
                           nprobe=100,
                           emb_size=768)
encoder_config = EncoderConfig(is_finetune=True,
                              doc_encoder_name_or_path='Shitao/RetroMAE_MSMARCO_distill',
                            query_encoder_name_or_path='Shitao/RetroMAE_MSMARCO_distill')

index = AutoIndex.get_index(index_config, encoder_config, data)
index.train(data=data,
            per_query_neg_num=1,
            per_device_train_batch_size=512,
            logging_steps=100,
            epochs=16)

index.save_all('./data/MSMARCO/parameters')
```

We can observe indicators by the following methods:

```python
# MRR and RECALL
from LibVQ.dataset.dataset import load_rel
if data.dev_queries_embedding_dir is not None:
	dev_query = index.get(data, data.dev_queries_embedding_dir)
	ground_truths = load_rel(data.dev_rels_path)
	index.test(dev_query, ground_truths, topk=1000, batch_size=64,
	MRR_cutoffs=[5, 10, 20, 50, 100], Recall_cutoffs=[5, 10, 20, 50, 100],
				nprobe=index_config.nprobe)
```

The result maybe follow:

```bash
number of query:6980,  searching time per query: 0.0005462420020882243
6980 matching queries found
MRR@5:0.3101456542502381
MRR@10:0.32432192886705746
MRR@20:0.33025472172271947
MRR@50:0.33296326894754924
MRR@100:0.3336563987436049
Recall@5:0.4641953199617957
Recall@10:0.5677531041069722
Recall@20:0.6505730659025789
Recall@50:0.7340019102196753
Recall@100:0.783440783190067
```

**The different result can be viewed as follow:**

**The normal result**：

index method: ivfpq

ivf_centers_num:10000

subvector_num:64

subvector_bits:8

nprobe:100

|                          | MRR@5  | MRR@20 | RECALL@5 | RECALL@20 | RECALL@100 |
| ------------------------ | ------ | ------ | -------- | --------- | ---------- |
| faiss                    | 0.2423 | 0.2627 | 0.3770   | 0.5713    | 0.7281     |
| contrastive              | 0.3067 | 0.3265 | 0.4619   | 0.6488    | 0.7811     |
| distill                  | 0.3101 | 0.3303 | 0.4642   | 0.6506    | 0.7834     |
| contrastive with encoder | 0.3223 | 0.3437 | 0.4908   | 0.6957    | 0.8419     |
| distill with encoder     | 0.3252 | 0.3471 | 0.4928   | 0.6979    | 0.8426     |

**The result of embedding through the pooler:**

index method: opq

subvector_num:64

subvector_bits:8

index_emb_size:512

data_emb_size:768

|                          | MRR@5  | MRR@20 | RECALL@5 | RECALL@20 | RECALL@100 |
| ------------------------ | ------ | ------ | -------- | --------- | ---------- |
| faiss                    | 0.3255 | 0.3463 | 0.4879   | 0.6843    | 0.8498     |
| contrastive              | 0.3454 | 0.3663 | 0.5172   | 0.7164    | 0.8694     |
| distill                  | 0.3460 | 0.3674 | 0.5178   | 0.7199    | 0.8737     |
| contrastive with encoder | 0.3637 | 0.3871 | 0.5398   | 0.7610    | 0.9070     |
| distill with encoder     | 0.3619 | 0.3848 | 0.5418   | 0.7613    | 0.9081     |

### search

To use the search function here, you must have built the index, and then pass in the save path of the parameters of index and query statement:

```python
from LibVQ.learnable_index import LearnableIndex
from LibVQ.dataset import Datasets

data = Datasets('./data/MSMARCO', emb_size=768)
index = LearnableIndex.load_all('./data/MSMARCO/parameters')
if data.docs_path is not None:
    query = ['what is paranoid sc', 'what is mean streaming']
    answer, answer_id = index.search_query(query, data)
    print(answer)
    print(answer_id)
```

The result maybe follow:

```bash
[['Paranoid schizophrenia Paranoid schizophrenia is the most common type of schizophrenia. Schizophrenia is defined as â\x80\x9ca chronic mental disorder in which a person loses touch with reality. Schizophrenia is divided into subtypes based on the â\x80\x9cpredominant symptomatology at the time of evaluation. The clinical picture is dominated by relatively stable and often persecutory delusions that are usually accompanied by hallucinations, particularly of the auditory variety, and perceptual disturbances. These symptoms ...', '- Paranoid schizophrenia is one of the 5 main subtypes of schizophrenia characterized by an intense paranoia which is often accompanied by delusions and hall Find this Pin and more on Abnormal psychology by rinkuatwal. Paranoid Schizophrenia: Symptoms, Causes, Treatment Paranoid Schizophrenia: Symptoms, Causes, Treatment See More', '- Paranoid schizophrenia, schizophrenia, paranoid type is a sub-type of schizophrenia as defined in the Diagnostic and Statistical Manual of Mental Disorders, DSM-IV code 295.30. It has been the most common type of schizophrenia.', "Paranoid Schizophrenia Paranoid Schizophrenia. Paranoid schizophrenia is the most common type of schizophrenia. It is characterized by prominent delusions that usually involve some form of threat or conspiracy, such as secret plots to control people's brains through radio transmissions.", 'Explore Abnormal Psychology and more! Paranoid schizophrenia is one of the 5 main subtypes of schizophrenia characterized by an intense paranoia which is often accompanied by delusions and hall.', 'Paranoid schizophrenia Paranoid schizophrenia is a lifelong disease, but with proper treatment, a person with the illness can attain a higher quality of life. [dead link] Although paranoid schizophrenia is defined by those two symptoms, it is also defined by a lack of certain symptoms (negative symptoms).', '- Paranoid schizophrenia is a subtype of schizophrenia in which the patient has delusions (false beliefs) that a person or some individuals are plotting against them or members of their family. Paranoid schizophrenia is the most common schizophrenia type.aranoid schizophrenia is a subtype of schizophrenia in which the patient has delusions (false beliefs) that a person or some individuals are plotting against them or members of their family. Paranoid schizophrenia is the most common schizophrenia type.', '- Paranoid Schizophrenia: Causes, Symptoms and Treatments. Paranoid schizophrenia is a subtype of schizophrenia in which the patient has delusions (false beliefs) that a person or some individuals are plotting against them or members of their family. Paranoid schizophrenia is the most common schizophrenia type.', 'Medications for Paranoid Disorder Medications for Paranoid Disorder. What is Paranoid Disorder: Paranoid personality disorder is a psychiatric condition in which a person has a long-term distrust and suspicion of others, but does not have a full-blown psychotic disorder such as schizophrenia. Compare drugs associated with Paranoid Disorder.', 'What is Paranoid Schizophrenia? Symptoms, Causes, Treatments Paranoid schizophrenia represents the most common of the many sub-types of the debilitating mental illness known collectively as schizophrenia. People with all types of schizophrenia become lost in psychosis of varying intensity, causing them to lose touch with reality.', 'What is Paranoid Schizophrenia? Symptoms, Causes, Treatments Patients often describe life with paranoid schizophrenia as a dark and fragmented world â\x80\x93 a life marked by suspicion and isolation where voices and visions torment them in a daily waking nightmare. Common paranoid schizophrenia symptoms may include:', 'Explore Abnormal Psychology and more! Paranoid schizophrenia is a psychotic disorder. In-depth information on symptoms, causes, treatment of paranoid schizophrenia. Find this Pin and more on Schizophrenia by healthyplace. Paranoid schizophrenia is a psychotic disorder. In-depth information on symptoms, causes, treatment of paranoid schizophrenia. See More', 'Paranoid schizophrenia Paranoid schizophrenia is the most common type of schizophrenia in most parts of the world. The clinical picture is dominated by relatively stable, often paranoid, delusions, usually accompanied by hallucinations, particularly of the auditory variety, and perceptual disturbances.aranoid schizophrenia is the most common type of schizophrenia in most parts of the world. The clinical picture is dominated by relatively stable, often paranoid, delusions, usually accompanied by hallucinations, particularly of the auditory variety, and perceptual disturbances.', 'paranoid paranoid disorder older term for delusional disorder. paranoid personality disorder a personality disorder in which the patient views other people as hostile, devious, and untrustworthy and reacts in a combative manner to disappointments or to events that he or she considers rebuffs or humiliations.', 'Paranoid Personality Disorder Paranoid personality disorder (PPD) is one of a group of conditions called Cluster A personality disorders which involve odd or eccentric ways of thinking. People with PPD also suffer from paranoia, an unrelenting mistrust and suspicion of others, even when there is no reason to be suspicious.', '- Paranoid schizophrenia is the most common type of schizophrenia. Schizophrenia is defined as â\x80\x9ca chronic mental disorder in which a person loses touch with reality . Schizophrenia is divided into subtypes based on the â\x80\x9cpredominant symptomatology at the time of evaluation.', 'Paranoia Paranoia is the irrational and persistent feeling that people are â\x80\x98out to get youâ\x80\x99. The three main types of paranoia include paranoid personality disorder, delusional (formerly paranoid) disorder and paranoid schizophrenia. Treatment aims to reduce paranoid and other symptoms and improve the personâ\x80\x99s ability to function.', "Paranoid schizophrenia Paranoid schizophrenia is differentiated by the presence of hallucinations and delusions involving the perception of persecution or grandiosity in one's beliefs about the world. People with paranoid schizophrenia are often more articulate or normal seeming than other people with schizophrenia, such as disorganized schizophrenia (hebephrenia)-afflicted individuals.", '- Paranoid personality disorder (paranoial psychopathy), F60.0: These people are inclined at stable overvalued ideas. These overvalued ideas may be of different types: ideas of persecution, jealousy, imitative arts, reformation, fabrication, hypochondria, dysmorphomania etc.', "What Is Paranoid Schizophrenia? Paranoid schizophrenia, or schizophrenia with paranoia as doctors now call it, is the most common example of this mental illness. Schizophrenia is a kind of psychosis; your mind doesn't agree with reality. It affects how you think and behave."], ['streaming Definition of streaming. : relating to or being the transfer of data (such as audio or video material) in a continuous stream especially for immediate processing or playback.', "What does streaming mean? Streaming is a generic term. It basically means that the data being transferred can be used immediately, without having to download the thing in it's entirety before it can be used. Streaming audio or video will be decoded and played back immediately -- or once enough data has been transferred in order to start playback. There are basically two types of streaming: When a media file is located on a traditional web server, it is served like other web-based files such as images, HTML and so on.", "What does streaming mean? What does streaming mean? Streaming is a generic term. It basically means that the data being transferred can be used immediately, without having to download the thing in it's entirety before it can be used. Streaming audio or video will be decoded and played back immediately -- or once enough data has been transferred in order to start playback.", "What does streaming mean? Streaming is a generic term. It basically means that the data being transferred can be used immediately, without having to download the thing in it's entirety before it can be used. Streaming audio or video will be decoded and played back immediately -- or once enough data has been transferred in order to start playback.", '- Definition of stream. intransitive verb. 1a : to flow in or as if in a streamb : to leave a bright trail a meteor streamed through the sky. 2a : to exude a bodily fluid profusely her eyes were streamingb : to become wet with a discharge of bodily fluid streaming with perspiration.', 'What is streaming? Streaming means listening to music or watching video in â\x80\x98real timeâ\x80\x99, instead of downloading a file to your computer and watching it later. With internet videos and webcasts of live events, there is no file to download, just a continuous stream of data.', 'streaming Streaming or media streaming is a technique for transferring data so that it can be processed as a steady and continuous stream. Streaming technologies are becoming increasingly important with the growth of the Internet because most users do not have fast enough access to download large multimedia files quickly. With streaming, the client browser or plug-in can start displaying the data before the entire file has been transmitted.', 'What is streaming? WebWise Team | 10th October 2012. Streaming means listening to music or watching video in â\x80\x98real timeâ\x80\x99, instead of downloading a file to your computer and watching it later. With internet videos and webcasts of live events, there is no file to download, just a continuous stream of data.', 'What does streaming mean? True Streaming. True streaming actually means that the data is played, then discarded immediately after it is played. Rather than delivering an entire file, streaming servers deliver mini-chunks of the source media. On the client side (your browser or media player) takes these little chunks of data and collects them so that, perhaps, 20 seconds of the media files can playback. This is referred to as buffering.', 'What is âStreamingâ and What Does it Mean? It isnâ\x80\x99t 100% accurate to think this, but it could help you while you work on getting your head around the term streaming. A proper definition of streaming is transmitting a continuous flow of audio and/or video data while earlier parts are being used.ownloading and streaming data are two contrasting ways of obtaining audio or video data. Looking back at our movie example, if a person downloads a copy of Clint Eastwoodâ\x80\x99s movie onto her computer so she can watch it again and again, then she is not streaming the movie when she watches it.', 'streaming Definition of streaming. 1  1 : the act, the process, or an instance of streaming data (see 2stream transitive 3) or of accessing data that is being streamed â\x80¦ among the hundreds whose presentations are freely available for streaming on the conference website. 2  2 : an act or instance of flowing; specifically : cyclosis. 3  3 British : tracking.', 'streaming Streaming or media streaming is a technique for transferring data so that it can be processed as a steady and continuous stream.Streaming technologies are becoming increasingly important with the growth of the Internet because most users do not have fast enough access to download large multimedia files quickly.or streaming to work, the client side receiving the data must be able to collect the data and send it as a steady stream to the application that is processing the data and converting it to sound or pictures.', 'stream Definition of stream. 1  intransitive verb. 2  1a : to flow in or as if in a streamb : to leave a bright trail a meteor streamed through the sky. 3  2a : to exude a bodily fluid profusely her eyes were streamingb : to become wet with a discharge of bodily fluid streaming with perspiration. 4  3 : to trail out at full length her hair streaming back as she ran.', 'Definitions &Translations Here are all the possible meanings and translations of the word streaming media. Freebase(0.00 / 0 votes)Rate this definition: Streaming media. Streaming media is multimedia that is constantly received by and presented to an end-user while being delivered by a provider. The verb to stream refers to the process of delivering media in this manner; the term refers to the delivery method of the medium, rather than the medium itself, and is an alternative to downloading.', 'Streaming media As of 2016, streaming is generally taken to refer to cases where a user watches digital video content and/or listens to digital audio content on a computer screen and speakers (ranging from a desktop computer to a smartphone) over the Internet.', 'Definitions &Translations Here are all the possible meanings and translations of the word streaming media. Freebase(0.00 / 0 votes)Rate this definition: Streaming media is multimedia that is constantly received by and presented to an end-user while being delivered by a provider.', 'What does the word video streaming means? You mean streaming video. It means as you watch it it will not be choppy eventhough it is constantly coming from the internet. All streaming videos come from the internet.', 'Definitions &Translations Definitions for streaming media. Here are all the possible meanings and translations of the word streaming media. Freebase(0.00 / 0 votes)Rate this definition: Streaming media is multimedia that is constantly received by and presented to an end-user while being delivered by a provider.', '- Definitions for streaming media. Here are all the possible meanings and translations of the word streaming media. Freebase(0.00 / 0 votes)Rate this definition: Streaming media. Streaming media is multimedia that is constantly received by and presented to an end-user while being delivered by a provider.', "What does the word video streaming means? It means that the video is coming from the internet as you watch it. When a video is downloaded, the entire file is brought onto your computer and properly stored before you watch it. In a streaming video the images are coming directly off the internet. If the video doesn't stream fast enough, the image may sometimes have to pause while the temporary data catches up (if this is happening to you, hit the pause button for a few minutes to let the data get a bit ahead of the video)."]]
[['1266114', '7187021', '3798', '1473032', '7187024', '1266113', '8768754', '3807', '7959743', '4379973', '4379979', '7187023', '4734246', '633649', '5714137', '1266112', '7959744', '1266120', '6782168', '4787078'], ['7183229', '1758914', '196677', '6707877', '1692116', '615518', '1758911', '1730796', '1758917', '321078', '1450137', '6191098', '1692125', '615515', '3493858', '615519', '8472643', '1450135', '1450134', '8472644']]
```

### embedding compression

To use the search function here, you must have built the pq/opq index(and you can only use pq/opq index), and then pass in the save path of the index

You can get quantization by this:

```python
from embedding_compression.quantization import Quantization

pq = Quantization.from_faiss_index('./data/MSMARCO/parameters/learnable.index')
```

Then we get the data to be compressed. Here, we use numpy random data

```python
>>> import numpy as np
>>> sample_embedding = np.memmap('test.memmap', dtype=np.float32, mode="w+", shape=(3,768))
>>> sample_np = np.random.random((3,768))
>>> sample_embedding[:] = sample_np[:]
>>> print(sample_embedding)
memmap([[0.96952295, 0.28536782, 0.43043545, ..., 0.25155088, 0.6586514 ,
         0.01303443],
        [0.17380722, 0.72505873, 0.8279453 , ..., 0.09535836, 0.9301384 ,
         0.5932224 ],
        [0.04763605, 0.46201319, 0.38883942, ..., 0.85009974, 0.07128737,
         0.990246  ]], dtype=float32)
```

If you want to get the compressed embedding, you can use the following statement

```python
>>> import torch
>>> print('The compressed results are as follows:')
>>> tensor_emb = torch.Tensor(sample_embedding)
>>> result = pq.embedding_compression(tensor_emb)
>>> print(result.shape)
torch.Size([3, 64])
>>> print(result)
tensor([[ 25,  85,  86, 186,  53, 220, 174,  77,  90, 100, 115,  22,  83, 201,
          70,  43, 240, 146, 210,  51, 215, 214,  44, 223, 252, 247, 244,  96,
         113, 214, 232, 138, 172, 205, 236, 170,   5, 114, 178,  94, 228, 156,
         149,   9, 141,  10,  74, 189, 144,  39,  58, 168,  32,  57, 237,  74,
           2,  59, 173, 118,  17, 121,  54, 244],
        [210,  53, 180, 186,   1,  27, 196,  83, 126, 130, 179,  84,  83, 139,
         254, 246, 100, 164, 240, 163, 159, 199, 201,  13, 119, 247,  84, 226,
          80, 137, 230,  30, 172, 205, 138, 170, 161, 243, 106,  84, 228, 132,
          51, 191,   7,  10,  74, 189, 133, 236,  44,  46,  48, 217, 255, 186,
         133,  56, 242, 166,  17, 103,  54, 239],
        [201,  53,  21, 186, 199,  58, 165,  77, 126, 130, 169,  22,  83, 174,
         106, 179, 144,   6, 119,  50, 159, 157, 201,  93, 222, 247,  84, 120,
          61, 187, 230,  50, 232, 213,  91, 170, 227, 114, 176,  30,  69, 147,
         191,   9, 199,  16, 234, 189, 201, 104,  41,  46,   9,  57,  55,  74,
           2, 238, 126, 242,  17, 254, 183, 239]])

```

If you want to get the result before compression, you can use the following statement

```python
>>> print('The nearest centroid of the vector is as follows:')
>>> centroid = pq.get_quantization_vecs(result)
>>> print(centroid.shape)
torch.Size([3, 768])
>>> print(centroid)
tensor([[ 0.2223,  0.2161, -0.0219,  ...,  0.0900,  0.3536, -0.0138],
        [ 0.1766,  0.0505, -0.0798,  ...,  0.1910, -0.0275,  0.2142],
        [-0.0463,  0.0782,  0.1338,  ...,  0.1910, -0.0275,  0.2142]])
```

### topic modeling

To use the search function here, you must have built the ivf index(ivf/ivf_pq/ivf_opq), and then pass in the save path of the index and collection path

You can get classes(combined document clustering results) by this:

```python
from topic_modeling.topic_model import TopicModel, get_classes

classes = get_classes('./data/MSMARCO/collection.tsv','./data/MSMARCO/parameters/learnable.index')
```

And then we can extracting topics:

```python
topic_model = TopicModel(classes)
```

After generating topics and their probabilities, we can access the frequent topics that were generated:

```python
>>> topic_model.get_all_topic_info()

     Topic	     Count	                Name
        -1	 195134563	    -1_the_of_and_to
         0	     11012	0_alice_wonderland_adventures_carroll
         1	     55056	1_meiosis_cells_cell_chromosomes
         2	     25266	2_weather_summer_july_august
         3	     23131	3_tube_catheter_tubing_bag
...................
      9996	     19076	9996_server_client_computer_proxy
      9997	     15292	9997_calcium_hypocalcemia_blood_low
      9998	     52765	9998_acid_base_acids_bases
      9999	     12422	9999_symbol_yang_yin_swastika
```

-1 refers to all outliers and should typically be ignored. 

```python
>>> topic_model.get_topic_info(0)

     Topic	     Count	                Name
         0	     11012	0_alice_wonderland_adventures_carroll
```

We use c_tf_idf in bertopic to compute the score.

Next, let's take a look at the topic 2 as an example:

```python
>>> topic_model.get_topic(2)

{'weather': 0.6736024042439646, 'summer': 0.5221852247045404, 'july': 0.4904021880695438, 'august': 0.4636609312538878, 'june': 0.3647692310882189, 'time': 0.27948571686385265, 'best': 0.2564795286433169, 'visit': 0.236850503863349, 'months': 0.23132943339407794, 'september': 0.2233210283479634}
```