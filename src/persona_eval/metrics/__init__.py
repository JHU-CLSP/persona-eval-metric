from persona_eval.metrics.base import get_metric, list_metrics, register_metric  # noqa: F401

# Import metric modules to trigger registration
from persona_eval.metrics import summeval_metrics  # noqa: F401  (SUPERT, SummaQA, BLANC)
from persona_eval.metrics import rouge_metrics  # noqa: F401
from persona_eval.metrics import bertscore_metric  # noqa: F401
from persona_eval.metrics import syntactic_metric  # noqa: F401
from persona_eval.metrics import llm_judge_metric  # noqa: F401
from persona_eval.metrics import factscore_metric  # noqa: F401
