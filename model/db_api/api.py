from functools import lru_cache
from typing import List

import pandas as pd

from model.utils.data_utils import DATA_METAINFO_FILE



def split_string(input_string: str, separator: str = " "):
    return input_string.split(separator)


def list_length(input_list: list):
    return len(input_list)


@lru_cache()
def get_paper_authors(paper_name: str) -> List[str]:
    df = pd.read_csv(DATA_METAINFO_FILE, delimiter='\t')
    if paper_name not in df['title'].to_list():
        return []
    author_names = df[df['title'] == paper_name]['authors'].to_list().pop()

    return author_names.split(',')


@lru_cache()
def get_paper_abstract(paper_name: str) -> List[str]:
    df = pd.read_csv(DATA_METAINFO_FILE, delimiter='\t')
    if paper_name not in df['title'].to_list():
        return []
    abstract = df[df['title'] == paper_name]['abstract'].to_list().pop()

    return abstract
