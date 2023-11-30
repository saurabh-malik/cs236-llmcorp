import csv
import functools
import os.path
from enum import Enum
from typing import List, Tuple

import pandas as pd


class DataSplitGroup(Enum):
    # Use the group index of each paper in `metainfo.tsv` to distinguish data category
    TRAIN = [0, 1, 2, 3, 4, 5, 6, 7]
    DEV = [8]
    TEST = [9]


DATA_METAINFO_FILE = relative_path = os.path.join(os.path.dirname(__file__), '../../data/papers/metainfo.tsv')


def get_paper_names_by_group_idx(idx: int) -> List[str]:
    df = pd.read_csv(DATA_METAINFO_FILE, delimiter='\t')

    papers_of_group = df[df['group'] == idx]

    all_paper_names = papers_of_group['title'].tolist()

    return all_paper_names


def get_paper_and_author_names_by_group_idx(idx: int) -> List[Tuple[str, List[str]]]:
    df = pd.read_csv(DATA_METAINFO_FILE, delimiter='\t')

    papers_of_group = df[df['group'] == idx]

    all_paper_names = papers_of_group['title'].tolist()
    all_paper_authors = papers_of_group['authors'].tolist()

    zipped_info = [(paper_name, paper_author.split(',')) for paper_name, paper_author in zip(all_paper_names, all_paper_authors)]

    return zipped_info


def get_paper_and_abstracts_by_group_idx(idx: int) -> List[Tuple[str, List[str]]]:
    df = pd.read_csv(DATA_METAINFO_FILE, delimiter='\t')

    papers_of_group = df[df['group'] == idx]

    all_paper_names = papers_of_group['title'].tolist()
    all_paper_abstracts = papers_of_group['abstract'].tolist()

    zipped_info = [(paper_name, paper_abstract) for paper_name, paper_abstract in zip(all_paper_names, all_paper_abstracts)]

    return zipped_info


def get_paper_names_by_split(split: DataSplitGroup) -> List[str]:
    return functools.reduce(lambda x, y: x+y, [get_paper_names_by_group_idx(idx) for idx in split.value], [])