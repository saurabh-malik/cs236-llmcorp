# Use tool (API call) to achieve task rather than simply relying on RAG
import json
import os
from abc import ABC
import datetime

import re
from collections import defaultdict

import torch
import transformers
import tqdm
from langchain.chains import ConversationChain
from rouge_score import rouge_scorer
from transformers import LlamaForCausalLM, LlamaTokenizer, Conversation, BitsAndBytesConfig

from config import AppConfig
from .evaluation_task import EvaluationTask
from model.utils.data_utils import DataSplitGroup, get_paper_and_author_names_by_group_idx, \
    get_paper_and_abstracts_by_group_idx
from model.utils.setup_utils import CONTEXT, get_llm
from model.db_api.api import *


class EvaluationTaskToolBase(EvaluationTask, ABC):
    def __init__(self, name: str, exp_dir: str, exp_split: DataSplitGroup):
        super().__init__(name=name)
        self.exp_dir = exp_dir
        self.exp_split = exp_split
        self.pipeline = None

        self.device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
        self.question_answers = None
        self.report = {}

    def setup(self):
        self.question_answers = self._prepare_test_set()
        os.makedirs(self.exp_dir)

        model = LlamaForCausalLM.from_pretrained(
            AppConfig.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        ).eval()
        tokenizer = LlamaTokenizer.from_pretrained(AppConfig.model_id)
        self.pipeline = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer,
                                              torch_dtype=torch.float16, device_map='auto')

        # llm = get_llm()
        # self.pipeline = ConversationChain(llm=llm)

    def _prepare_test_set(self):
        raise NotImplementedError

    def dump_result_json(self):
        with open(f"{self.exp_dir}/report.json", 'w') as writer:
            json.dump(self.report, writer, indent=4)


class AuthorNameEvaluationTool(EvaluationTaskToolBase):
    def _get_result(self, question: str) -> str:
        # conversation = Conversation(CONTEXT + "\n" + f"now please answer question: {question}", max_length=500)
        prompt = CONTEXT + "\n" + f"Question: {question}\nAnswer: "
        conversation = self.pipeline(prompt)
        completed_text = conversation[0]['generated_text']
        answer = completed_text[len(prompt):]
        return answer

    def run(self):
        # TODO: this can be optimized by doing batch inference
        answers = []
        total_runtime = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    answers.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    answers.append('torch.cuda.OutOfMemoryError')

        correct_call = 0
        failed_call = 0
        partial_call = 0
        error_count = 0
        total_count = 0
        results = []
        for direct_answer, (question, ref_authors) in zip(answers, self.question_answers):
            # 2. log details
            # convert data in results to string
            result = dict()
            result['question'] = question
            result['ref_authors'] = ref_authors
            result['direct_answer'] = direct_answer

            # 1. accumulate counts
            total_count += 1
            if 'error' in result:
                error_count += 1
                result['result'] = 'ERROR'
                continue

            try:
                python_command = re.search(r".*<begin_call>(.*)<end_call>.*", direct_answer).groups()[0]
                eval_result = eval(python_command)
                if all(author_name in eval_result for author_name in ref_authors):
                    correct_call += 1
                    result['result'] = 'SUCCESS'
                elif any(author_name in eval_result for author_name in ref_authors):
                    partial_call += 1
                    result['result'] = 'PARTIAL'
                else:
                    failed_call += 1
                    result['result'] = 'FAIL'
            except:
                error_count += 1
                result['result'] = 'ERROR'

            results.append(result)

        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        self.report = {
            'total': total_count,
            'correct_func_call': correct_call,
            'partial_call': partial_call,
            'failed_call': failed_call,
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


class AuthorCountEvaluationTool(EvaluationTaskToolBase):
    def _get_result(self, question: str) -> str:
        # conversation = Conversation(CONTEXT + "\n" + f"now please answer question: {question}", max_length=500)
        prompt = CONTEXT + "\n" + f"Question: {question}\nAnswer: "
        conversation = self.pipeline(prompt)
        completed_text = conversation[0]['generated_text']
        answer = completed_text[len(prompt):]
        return answer

    def run(self):
        # TODO: this can be optimized by doing batch inference
        answers = []
        total_runtime = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    answers.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    answers.append('torch.cuda.OutOfMemoryError')

        correct_call = 0
        failed_call = 0
        format_error_count = 0
        error_count = 0
        total_count = 0
        results = []
        for direct_answer, (question, ref_authors) in zip(answers, self.question_answers):
            # 2. log details
            # convert data in results to string
            result = dict()
            result['question'] = question
            result['ref_authors'] = ref_authors
            result['direct_answer'] = direct_answer

            # 1. accumulate counts
            total_count += 1
            if 'error' in result:
                error_count += 1
                result['result'] = 'ERROR'
                continue

            try:
                python_command = re.search(r".*<begin_call>(.*)<end_call>.*", direct_answer).groups()[0]
                eval_result = eval(python_command)
                number = re.search(r'\d+', eval_result)
                if number is None:
                    result['result'] = 'FORMAT ERROR'
                    format_error_count += 1
                else:
                    if int(number[0]) == len(ref_authors):
                        result['result'] = 'CORRECT NUMBER'
                        correct_call += 1
                    else:
                        result['result'] = 'WRONG NUMBER'
                        failed_call += 1
            except:
                error_count += 1
                result['result'] = 'ERROR'

            results.append(result)

        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        self.report = {
            'total': total_count,
            'correct_func_call': correct_call,
            'format_error_count': format_error_count,
            'failed_call': failed_call,
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
            question = f"How many authors are there in paper {paper_name}?"
            question_answers.append((question, author_names))

        return question_answers


class AuthorEvenOddEvaluationTool(EvaluationTaskToolBase):
    def _get_result(self, question: str) -> str:
        # conversation = Conversation(CONTEXT + "\n" + f"now please answer question: {question}", max_length=500)
        prompt = CONTEXT + "\n" + f"Question: {question}\nAnswer: "
        conversation = self.pipeline(prompt)
        completed_text = conversation[0]['generated_text']
        answer = completed_text[len(prompt):]
        return answer

    def run(self):
        # TODO: this can be optimized by doing batch inference
        answers = []
        total_runtime = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    answers.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    answers.append('torch.cuda.OutOfMemoryError')

        correct_call = 0
        failed_call = 0
        format_error_count = 0
        error_count = 0
        total_count = 0
        results = []
        for direct_answer, (question, ref_authors) in zip(answers, self.question_answers):
            # 2. log details
            # convert data in results to string
            result = dict()
            result['question'] = question
            result['ref_authors'] = ref_authors
            result['direct_answer'] = direct_answer

            # 1. accumulate counts
            total_count += 1
            if 'error' in result:
                error_count += 1
                result['result'] = 'ERROR'
                continue

            try:
                python_command = re.search(r".*<begin_call>(.*)<end_call>.*", direct_answer).groups()[0]
                eval_result = eval(python_command)
                eval_result = eval_result.lower()
                correct_answer = 'even' if len(ref_authors) % 2 == 0 else 'odd'
                wrong_answer = 'odd' if len(ref_authors) % 2 == 0 else 'even'

                if correct_answer in eval_result and wrong_answer in eval_result:
                    format_error_count += 1
                elif correct_answer in eval_result:
                    correct_call += 1
                elif wrong_answer in eval_result:
                    failed_call += 1
                else:
                    format_error_count += 1
            except:
                error_count += 1
                result['result'] = 'ERROR'

            results.append(result)


        with open(f"{self.exp_dir}/results.json", 'w') as writer:
            json.dump(results, writer, indent=4)

        self.report = {
            'total': total_count,
            'correct_func_call': correct_call,
            'format_error_count': format_error_count,
            'failed_call': failed_call,
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
            question = f"Are there even number or odd number of authors in paper {paper_name}?"
            question_answers.append((question, author_names))

        return question_answers


class SummaryEvaluationTool(EvaluationTaskToolBase):
    def _get_result(self, question: str) -> str:
        # conversation = Conversation(CONTEXT + "\n" + f"now please answer question: {question}", max_length=500)
        prompt = CONTEXT + "\n" + f"Question: {question}\nAnswer: "
        conversation = self.pipeline(prompt)
        completed_text = conversation[0]['generated_text']
        answer = completed_text[len(prompt):]
        return answer

    def run(self):
        # TODO: this can be optimized by doing batch inference
        answers = []
        total_runtime = 0
        error_count = 0
        with torch.no_grad():
            for question, _ in tqdm.tqdm(self.question_answers):
                try:
                    start_runtime = datetime.datetime.now()
                    answers.append(self._get_result(question))
                    end_runtime = datetime.datetime.now()
                    total_runtime += (end_runtime - start_runtime).total_seconds()
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning(f"OutOfMemoryError on {question}: skipping")
                    answers.append('torch.cuda.OutOfMemoryError')

        scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
        scores_total = defaultdict(float)
        results = []
        for direct_answer, (question, ref_abstract) in zip(answers, self.question_answers):
            # 2. log details
            # convert data in results to string
            result = dict()
            result['question'] = question
            result['ref_abstract'] = ref_abstract
            result['direct_answer'] = direct_answer

            # 1. accumulate counts
            if 'error' in result:
                error_count += 1
                result['result'] = 'ERROR'
                continue

            try:
                python_command = re.search(r".*<begin_call>(.*)<end_call>.*", direct_answer).groups()[0]
                eval_result = eval(python_command)
                rouge_scores = scorer.score(ref_abstract, eval_result)
                result.update(rouge_scores)

                for rouge_score_name in rouge_scores:
                    scores_total[rouge_score_name] += rouge_scores[rouge_score_name].fmeasure
            except:
                result['result'] = 'ERROR'

            results.append(result)

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

    def _prepare_test_set(self):
        all_paper_abstracts = []
        for paper_group_idx in self.exp_split.value:
            paper_abstracts = get_paper_and_abstracts_by_group_idx(paper_group_idx)
            all_paper_abstracts.extend(paper_abstracts)

        question_answers = []
        for paper_name, author_names in all_paper_abstracts:
            question = f"Please summarize paper \"{paper_name}\""
            question_answers.append((question, author_names))

        return question_answers
