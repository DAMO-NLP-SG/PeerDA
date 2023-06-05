import sys
sys.path.append("..")
from types import SimpleNamespace
from task-TC_scorer import main

def test_input_submission_format():
    args = SimpleNamespace(submission="", gold="data", log_file="ciao.log", propaganda_techniques_list_file="")

