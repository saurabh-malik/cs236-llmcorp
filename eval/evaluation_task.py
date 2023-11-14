import abc
import json
import os

import torch
import tqdm

from model.customchain.chains import MyConversationalRetrievalChain
from model.utils.data_utils import DataSplitGroup, get_paper_and_author_names_by_group_idx
from model.utils.logging_utils import activate_logger
from model.utils.setup_utils import get_llm, get_vector_db_on_split


class EvaluationTask(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.logger = activate_logger(name)

    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractmethod
    def dump_result_json(self):
        raise NotImplementedError


class AuthorNameEvaluation(EvaluationTask):
    def __init__(self, name: str, exp_dir: str, exp_split: DataSplitGroup,
                 chain: MyConversationalRetrievalChain = None):
        super().__init__(name=name)
        self.exp_dir = exp_dir
        self.exp_split = exp_split
        self.chain = chain

        self.device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
        self.question_answers = None
        self.report = {}

    def setup(self):
        self.question_answers = self._prepare_author_name_test_set()
        os.makedirs(self.exp_dir)

        # Set up LLM, vector DB and the lang chain
        if self.chain is not None:
            self.logger.info(f"Use pre-constructed chain")
            return

        llm = get_llm()  # Obtain the MyHuggingFacePipeline object from the dependency
        vector_db = get_vector_db_on_split(self.exp_split)  # Obtain the MyFAISS object from the dependency

        self.chain = MyConversationalRetrievalChain.from_llm(llm.pipeline, vector_db.as_retriever(),
                                                             return_source_documents=True)

    def run(self):
        # TODO: this can be optimized by doing batch inference
        results = []
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    results.append(self.chain({"question": question, "chat_history": []}))
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    results.append({'question': question, 'error': 'torch.cuda.OutOfMemoryError'})

        all_author_correct = 0
        some_author_correct = 0
        none_author_correct = 0
        error_count = 0
        total_count = 0
        for result, (question, ref_authors) in zip(results, self.question_answers):
            # 2. log details
            # convert data in results to string
            for k, v in result.items():
                result[k] = str(v)
            result['ref_authors'] = ref_authors

            # 1. accumulate counts
            total_count += 1
            if 'error' in result:
                error_count += 1
                result['result'] = 'ERROR'
                continue

            direct_answer = result['answer']
            if all([author_name.lower() in direct_answer.lower() for author_name in ref_authors]):
                all_author_correct += 1
                result['result'] = 'SUCCESS'
            elif all([author_name.lower() not in direct_answer.lower() for author_name in ref_authors]):
                none_author_correct += 1
                result['result'] = 'FAIL'
            else:
                some_author_correct += 1
                result['result'] = 'PARTIAL'

        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        self.report = {
            'total': total_count,
            'all_correct': all_author_correct,
            'some_author_correct': some_author_correct,
            'non_author_correct': none_author_correct,
            'error_count': error_count
        }

    def dump_result_json(self):
        with open(f"{self.exp_dir}/report.json", 'w') as writer:
            json.dump(self.report, writer, indent=4)

    def _prepare_author_name_test_set(self):
        all_paper_author_names = []
        for paper_group_idx in self.exp_split.value:
            paper_author_names = get_paper_and_author_names_by_group_idx(paper_group_idx)
            all_paper_author_names.extend(paper_author_names)

        question_answers = []
        for paper_name, author_names in all_paper_author_names:
            question = f"Who are the authors of paper {paper_name}?"
            question_answers.append((question, author_names))

        return question_answers

