from eval.evaluation_task import AuthorNameEvaluation
from model.utils.data_utils import DataSplitGroup

# 0. Configure the splits that will be used for evaluation
EXP_SPLIT = DataSplitGroup.TEST
EXP_RESULT_DIR = "eval_results/"
EXP_NAME = "author_name_qa_test"

# Create Evaluation task and run
eval_task = AuthorNameEvaluation(EXP_NAME, EXP_RESULT_DIR, EXP_SPLIT)
eval_task.setup()
eval_task.run()
eval_task.dump_result_json()