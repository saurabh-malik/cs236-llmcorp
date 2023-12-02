import os

from eval.evaluation_task import (AuthorNameEvaluation, AuthorNameEvaluationNoRAG, SummaryEvaluation,
                                  AuthorCountEvaluation, AuthorEvenOddEvaluation)
from model.utils.data_utils import DataSplitGroup

# 0. Configure the splits that will be used for evaluation
EXP_SPLIT = DataSplitGroup.TEST
EXP_RESULT_DIR = "experiments/author_count_eval_results_llama2_7b/"
EXP_NAME = "author_count_qa_test"
# Create Evaluation task and run
eval_task = AuthorCountEvaluation(EXP_NAME, EXP_RESULT_DIR, EXP_SPLIT)
eval_task.setup()
eval_task.run()
eval_task.dump_result_json()

# EXP_RESULT_DIR_NO_RAG = "eval_results_no_rag_llama2_13b/"
# EXP_NAME_NO_RAG = "author_name_qa_test_no_rag"
# # Create Evaluation task and run
# eval_task = AuthorNameEvaluationNoRAG(EXP_NAME_NO_RAG, EXP_RESULT_DIR_NO_RAG, EXP_SPLIT)
# eval_task.setup()
# eval_task.run()
# eval_task.dump_result_json()
