import csv
import logging

import regex
import unicodedata
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

logger = logging.getLogger()


def check_answer(passages, answers, doc_ids, tokenizer):
    """Search through all the top docs to see if they have any of the answers."""
    hits = []
    for i, doc_id in enumerate(doc_ids):
        text = passages[doc_id][0]
        hits.append(has_answer(answers, text, tokenizer))
    return hits


def has_answer(answers, text, tokenizer, match_type='string') -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def _normalize(text):
    return unicodedata.normalize('NFD', text)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]


def load_test_data(query_andwer_file, collections_file):
    questions = []
    answers = []
    for line in open(query_andwer_file, encoding='utf-8'):
        line = line.strip().split('\t')
        questions.append(line[0])
        answers.append(eval(line[1]))

    collections = {}
    for line in open(collections_file, encoding='utf-8'):
        line = line.strip().split('\t')
        collections[int(line[0])] = (line[1], line[2])
    return questions, answers, collections



def validate(ann_items, questions, answers, collections):
    # print(ann_items)
    v_dataset = V_dataset(ann_items, questions, answers, collections)
    v_dataloader = DataLoader(v_dataset, batch_size=128, shuffle=False, num_workers=24, collate_fn=DataCollator())
    # print(len(ann_items), len(questions), len(answers), len(collections))

    final_scores = []
    for k, scores in enumerate(tqdm(v_dataloader, total=len(v_dataloader))):
        # print(scores)
        final_scores.extend(scores)
    # print(len(final_scores))

    n_docs = len(ann_items[0])
    top_k_hits = [0] * n_docs
    for question_hits in final_scores:
        # print(len(question_hits))

        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        # print(question_hits, best_hit)
        if best_hit is not None:
            # print(question_hits, best_hit)
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    top_k_hits = [v / len(ann_items) for v in top_k_hits]

    print(
        f"top5:{top_k_hits[4]}, top10:{top_k_hits[9]}, top20:{top_k_hits[19]}, top30:{top_k_hits[29]}, top50:{top_k_hits[49]}, top100:{top_k_hits[99]}")


class V_dataset(Dataset):
    def __init__(self,
                 ann_items, questions, answers, collections
                 ):
        # print("ann_items", ann_items)
        # print("ans", answers)
        self.questions = questions
        self.collections = collections
        self.answers = answers
        self.ann_items = ann_items
        tok_opts = {}
        self.tokenizer = SimpleTokenizer(**tok_opts)

    def __getitem__(self, query_id):
        doc_ids = [pidx for pidx in self.ann_items[query_id]]
        hits = []
        for i, doc_id in enumerate(doc_ids):
            if doc_id == -1:
                hits.append(False)
                title, text = '', ''
            else:
                title, text = self.collections[doc_id]
                hits.append(has_answer(self.answers[query_id], text, self.tokenizer))
        return hits

    def __len__(self):
        return len(self.ann_items)


class DataCollator():
    def __call__(self, batch_hits):
        return [x for x in batch_hits]
