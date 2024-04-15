import sys
sys.path.insert(0, '.')
import llm_common_eval as lce

questions = lce.mt_bench()
print(questions)
print(questions[0])
