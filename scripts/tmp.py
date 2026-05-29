import pandas as pd
import json
import random

# fname = "/scratch/mdredze1/icachol1/persona-eval/eval-results/run_20260428_162308/llm_responses/llm_responses_20260428_162311.jsonl"

# lines = [
#     json.loads(line) for line in open(fname, "r").readlines()
# ]
# df = pd.DataFrame(lines)
# print(df['metric'].unique())

# annotator = df[df['metric'] == 'llm_judge_annotator']

# i = random.randint(0, len(annotator)-1)

# print(annotator.iloc[i]['prompt'])
# print(annotator.iloc[i]['response'])
# print()


# fname = "/scratch/mdredze1/icachol1/persona-eval/eval-results/run_20260429_154722/metric_scores_position_bias.csv"
# df = pd.read_csv(fname)
# print(df['consistent'].value_counts())


fname = "/scratch/mdredze1/icachol1/persona-eval/robustness_results/perturbations/perturbations.csv"

df = pd.read_csv(fname)

print(df['dataset'].value_counts())