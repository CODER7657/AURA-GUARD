Model `openai/gpt-oss-120b` on 40 pre-registered questions (questions commit `af66b22`, code commit `0eb4fdf`).

| Metric | Before the checker (first draft) | After the checker (returned answer) |
|---|---|---|
| Grounded: every number, station and place traceable to a tool output | 30/40 (75%) | 40/40 (100%) |
| Expected behaviour (graded by string rules) | 33/40 (82%) | 32/40 (80%) |
| Correct refusal, among questions that should be refused | 11/12 (92%) | 11/12 (92%) |
| Refused although answerable | 5/28 (18%) | 6/28 (21%) |

Returned answers: 30 first drafts, 8 retries after checker feedback, 2 template fallbacks. API errors: 0.

Tool-call accuracy (all pre-registered calls made, with the right station and horizon): 21/23 (91%). Mean tool calls per question: 1.4.

Latency per question, end to end including retries: p50 32.8 s, p95 80.3 s; excluding time spent waiting on API rate limits: p50 2.0 s, p95 4.2 s.

| Category | Questions | Grounded before | Grounded after | Behaviour before | Behaviour after |
|---|---|---|---|---|---|
| answerable | 14 | 13/14 (93%) | 14/14 (100%) | 13/14 (93%) | 13/14 (93%) |
| ambiguous | 8 | 8/8 (100%) | 8/8 (100%) | 6/8 (75%) | 6/8 (75%) |
| out of scope | 10 | 5/10 (50%) | 10/10 (100%) | 9/10 (90%) | 9/10 (90%) |
| wrong premise | 8 | 4/8 (50%) | 8/8 (100%) | 5/8 (62%) | 4/8 (50%) |

Why first drafts failed the checker (a draft can fail for several reasons):

- numbers not found in any tool output: 7
- names Texas, where there is no monitor, though the question did not: 2
- names Washington, where there is no monitor, though the question did not: 1
- station id 48-201-1039-1 does not exist: 1
- names Colorado, where there is no monitor, though the question did not: 1

Returned answers graded wrong: A05 (answerable), B05 (ambiguous), B07 (ambiguous), C09 (out_of_scope), D01 (wrong_premise), D02 (wrong_premise), D03 (wrong_premise), D06 (wrong_premise)
