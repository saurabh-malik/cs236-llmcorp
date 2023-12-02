import abc
import json
import os
import re
from collections import defaultdict

import torch
import tqdm
import datetime

from rouge_score import rouge_scorer

from model.customchain.chains import MyConversationalRetrievalChain
from model.utils.data_utils import DataSplitGroup, get_paper_and_author_names_by_group_idx, \
    get_paper_and_abstracts_by_group_idx
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


class EvaluationTaskBase(EvaluationTask, abc.ABC):
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
        self.question_answers = self._prepare_test_set()
        os.makedirs(self.exp_dir)

        # Set up LLM, vector DB and the lang chain
        if self.chain is not None:
            self.logger.info(f"Use pre-constructed chain")
            return

        llm = get_llm()  # Obtain the MyHuggingFacePipeline object from the dependency
        vector_db = get_vector_db_on_split(self.exp_split)  # Obtain the MyFAISS object from the dependency

        self.chain = MyConversationalRetrievalChain.from_llm(llm.pipeline, vector_db.as_retriever(),
                                                             return_source_documents=True)

    def _prepare_test_set(self):
        raise NotImplementedError

    def dump_result_json(self):
        with open(f"{self.exp_dir}/report.json", 'w') as writer:
            json.dump(self.report, writer, indent=4)


class AuthorNameEvaluation(EvaluationTaskBase):
    def _get_result(self, question):
        return self.chain({"question": question, "chat_history": []})

    def run(self):
        # TODO: this can be optimized by doing batch inference
        results = []
        total_runtime = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    results.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
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
            'error_count': error_count,
            'total_runtime': total_runtime
        }

    def _prepare_test_set(self):
        all_paper_author_names = []
        for paper_group_idx in self.exp_split.value:
            paper_author_names = get_paper_and_author_names_by_group_idx(paper_group_idx)
            all_paper_author_names.extend(paper_author_names)

        question_answers = []
        for paper_name, author_names in all_paper_author_names:
            question = f"Who are the authors of paper {paper_name}?"
            question_answers.append((question, author_names))

        return question_answers


class AuthorNameEvaluationNoRAG(AuthorNameEvaluation):
    def __init__(self, name: str, exp_dir: str, exp_split: DataSplitGroup):
        super().__init__(name=name, exp_dir=exp_dir, exp_split=exp_split)
        self.llm = None

    def setup(self):
        self.question_answers = self._prepare_author_name_test_set()
        os.makedirs(self.exp_dir)

        self.llm = get_llm()  # Obtain the MyHuggingFacePipeline object from the dependency

    def _get_result(self, question):
        return {
            'answer': self.llm.prompt(question)
        }


class SummaryEvaluation(EvaluationTaskBase):
    def _prepare_test_set(self):
        all_paper_abstracts = []

        for paper_group_idx in self.exp_split.value:
            paper_abstracts = get_paper_and_abstracts_by_group_idx(paper_group_idx)
            all_paper_abstracts.extend(paper_abstracts)

        question_answers = []
        for paper_name, abstract in all_paper_abstracts:
            question = f"Please summarize the paper {paper_name} in 100 to 200 words."
            question_answers.append((question, abstract))

        return question_answers

    def _get_result(self, question):
        return self.chain({"question": question, "chat_history": []})

    def run(self):
        # TODO: this can be optimized by doing batch inference
        results = []
        total_runtime = 0
        error_count = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    results.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    results.append({'question': question, 'error': 'torch.cuda.OutOfMemoryError'})
                    error_count += 1

        scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
        scores_total = defaultdict(float)
        for result, (question, ref_abstract) in zip(results, self.question_answers):
            for k, v in result.items():
                result[k] = str(v)
            result['ref_abstract'] = ref_abstract

            direct_answer = result['answer']
            rouge_scores = scorer.score(ref_abstract, direct_answer)
            result.update(rouge_scores)

            for rouge_score_name in rouge_scores:
                scores_total[rouge_score_name] += rouge_scores[rouge_score_name].fmeasure

        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        effective_test_number = len(self.question_answers) - error_count
        average_scores = dict([(name, score / effective_test_number) for name, score in scores_total.items()])
        self.report = {
            'total': len(self.question_answers),
            'error_count': error_count,
            'total_runtime': total_runtime
        }
        self.report.update(average_scores)


class AuthorCountEvaluation(AuthorNameEvaluation):
    def _prepare_test_set(self):
        all_paper_author_names = []
        for paper_group_idx in self.exp_split.value:
            paper_author_names = get_paper_and_author_names_by_group_idx(paper_group_idx)
            all_paper_author_names.extend(paper_author_names)

        question_answers = []
        for paper_name, author_names in all_paper_author_names:
            question = f"How many authors is {paper_name} written by? Please answer with one arabic number."
            question_answers.append((question, author_names))

        return question_answers

    def run(self):
        # TODO: this can be optimized by doing batch inference
        results = []
        total_runtime = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    results.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    results.append({'question': question, 'error': 'torch.cuda.OutOfMemoryError'})

        correct_count = 0
        wrong_count = 0
        format_error_count = 0
        total_count = 0
        error_count = 0
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
            number = re.search(r'\d+', direct_answer)
            if number is None:
                result['result'] = 'FORMAT ERROR'
                format_error_count += 1
            else:
                if int(number[0]) == len(ref_authors):
                    result['result'] = 'CORRECT NUMBER'
                    correct_count += 1
                else:
                    result['result'] = 'WRONG NUMBER'
                    wrong_count += 1

        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        self.report = {
            'total': total_count,
            'correct_count': correct_count,
            'wrong_count': wrong_count,
            'format_error': format_error_count,
            'runtime_error_count': error_count,
            'total_runtime': total_runtime,
        }


class AuthorEvenOddEvaluation(AuthorNameEvaluation):
    def _prepare_test_set(self):
        all_paper_author_names = []
        for paper_group_idx in self.exp_split.value:
            paper_author_names = get_paper_and_author_names_by_group_idx(paper_group_idx)
            all_paper_author_names.extend(paper_author_names)

        question_answers = []
        for paper_name, author_names in all_paper_author_names:
            question = f"Are there even or odd number of authors who wrote {paper_name}? " \
                       f"Please answer 'even' or 'odd'."
            question_answers.append((question, author_names))

        return question_answers

    def run(self):
        # TODO: this can be optimized by doing batch inference
        results = []
        total_runtime = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    results.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    results.append({'question': question, 'error': 'torch.cuda.OutOfMemoryError'})

        correct_count = 0
        wrong_count = 0
        format_error_count = 0
        total_count = 0
        error_count = 0
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

            direct_answer = result['answer'].lower()
            correct_answer = 'even' if len(ref_authors) % 2 == 0 else 'odd'
            wrong_answer = 'odd' if len(ref_authors) % 2 == 0 else 'even'

            if correct_answer in direct_answer and wrong_answer in direct_answer:
                format_error_count += 1
            elif correct_answer in direct_answer:
                correct_count += 1
            elif wrong_answer in direct_answer:
                wrong_count += 1
            else:
                format_error_count += 1

        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        self.report = {
            'total': total_count,
            'correct_count': correct_count,
            'wrong_count': wrong_count,
            'format_error': format_error_count,
            'runtime_error_count': error_count,
            'total_runtime': total_runtime,
        }
