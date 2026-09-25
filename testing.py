from eval.groundedness_judge import judge_answer

from config import settings
print(settings.groq_model)

result = judge_answer(
    question="Forecast demand for TBP-001 next 3 months",
    context="Some sample context here.",
    answer="Demand for TBP-001 is expected to be around 300 units.",
)
print(result)