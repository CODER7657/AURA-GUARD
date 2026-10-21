Model `openai/gpt-oss-120b` on 19 pre-registered questions (questions commit `af66b22`, code commit `0eb4fdf`).

| Metric | Before the checker (first draft) | After the checker (returned answer) |
|---|---|---|
| Grounded: every number, station and place traceable to a tool output | 18/19 (95%) | 19/19 (100%) |
| Expected behaviour (graded by string rules) | 17/19 (89%) | 17/19 (89%) |
| Correct refusal, among questions that should be refused | n/a | n/a |
| Refused although answerable | 1/19 (5%) | 1/19 (5%) |

Returned answers: 18 first drafts, 1 retries after checker feedback, 0 template fallbacks. API errors: 0.

Tool-call accuracy (all pre-registered calls made, with the right station and horizon): 17/17 (100%). Mean tool calls per question: 1.7.

Latency per question, end to end including retries: p50 34.8 s, p95 405.5 s; excluding time spent waiting on API rate limits: p50 2.0 s, p95 3.5 s.

| Category | Questions | Grounded before | Grounded after | Behaviour before | Behaviour after |
|---|---|---|---|---|---|
| answerable | 14 | 13/14 (93%) | 14/14 (100%) | 13/14 (93%) | 13/14 (93%) |
| ambiguous | 5 | 5/5 (100%) | 5/5 (100%) | 4/5 (80%) | 4/5 (80%) |

Why first drafts failed the checker (a draft can fail for several reasons):

- numbers not found in any tool output: 1

Returned answers graded wrong: A05 (answerable), B05 (ambiguous)
