import sys
sys.path.insert(0, '.')
import llm_common_eval as lce

problems = lce.human_eval()
print(problems)
print(problems[0])
