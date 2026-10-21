# NO₂ forecaster

Forecasts hourly NO₂ at US monitoring stations 1, 6, 12 and 24 hours ahead, and
measures how well it does with a walk-forward backtest on real data.

## Data

[EPA AirData](https://aqs.epa.gov/aqsweb/airdata/download_files.html) hourly NO₂ for
2024 (`hourly_42602_2024.zip`, 29 MB, public, no account needed). The 20 monitors
with the most valid hours are used, one instrument per site: the Los Angeles basin,
Chicago, St. Louis, Kansas City, Phoenix, New Jersey, North Dakota, Utah and rural
Kansas, averaging from 1.3 to 26 ppb (`results/backtest_summary.json` lists them).
The same stations in the 2025 and 2026 files are used to check the trained model on
data it never saw and to issue the served forecasts.

## Method

A forecast is made at an origin hour *t* for the target hour *t + h*.

- **Features**, all from readings at or before *t*: the last 24 hourly values, the
  target hour one day and one week earlier, 6 h and 24 h rolling means, 24 h rolling
  spread, plus the target hour's local hour of day and weekday. Missing hours stay
  missing; nothing is filled in. A test checks that changing future readings changes
  no feature.
- **Models:** three baselines that any useful model has to beat (persistence, same
  hour yesterday, climatology by station, hour and weekday), ridge regression on the
  features, and histogram gradient boosting with the station as a categorical feature.
  One model per horizon.
- **Backtest:** for each month from July to December 2024, models are trained only on
  targets more than 14 days before that month, those last 14 days calibrate
  split-conformal 90% prediction intervals, and the month itself is the test set.
  Every number below is from test months the model never saw.

## Results

Pooled over all six test months (about 87,000 forecasts per horizon):

| Horizon | Model | MAE (ppb) | RMSE (ppb) | R² | vs persistence | 90% interval coverage |
|---|---|---|---|---|---|---|
| 1 h | Persistence | 2.00 | 3.43 | 0.904 | 0% | 88.8% |
| 1 h | Same hour yesterday | 4.85 | 7.77 | 0.506 | −143% | 88.2% |
| 1 h | Climatology | 5.16 | 8.01 | 0.475 | −158% | 87.2% |
| 1 h | Ridge on lags | 1.93 | 3.19 | 0.917 | +4% | 88.8% |
| 1 h | **Gradient boosting** | **1.85** | **3.13** | **0.920** | **+7%** | 88.4% |
| 6 h | Persistence | 5.30 | 8.28 | 0.439 | 0% | 88.1% |
| 6 h | Ridge on lags | 3.97 | 6.00 | 0.705 | +25% | 87.8% |
| 6 h | **Gradient boosting** | **3.69** | **5.81** | **0.724** | **+30%** | 87.6% |
| 12 h | Persistence | 6.11 | 9.16 | 0.312 | 0% | 88.5% |
| 12 h | Ridge on lags | 4.32 | 6.44 | 0.659 | +29% | 87.8% |
| 12 h | **Gradient boosting** | **4.11** | **6.41** | **0.663** | **+33%** | 87.5% |
| 24 h | Persistence | 4.88 | 7.79 | 0.504 | 0% | 88.2% |
| 24 h | Ridge on lags | 4.49 | 6.71 | 0.632 | +8% | 87.8% |
| 24 h | **Gradient boosting** | **4.32** | 6.74 | 0.628 | **+11%** | 87.6% |

All five models at every horizon are in [`results/summary.md`](results/summary.md).

![Forecast error by horizon](results/mae_by_horizon.png)

**What the numbers say**

- **R² on its own is misleading here.** Persistence, which predicts that the next hour
  equals this hour, already scores R² 0.90 one hour ahead. Skill against persistence
  is the honest measure of what a model adds.
- **The models earn most at 6 to 12 hours,** where NO₂ follows the daily traffic cycle
  and "same as now" is the wrong phase of it. At 24 h the same hour yesterday is already
  a strong guess, and the gain shrinks to 11%.
- **Intervals are slightly narrow:** 87.5–88.4% of outcomes fall inside the nominal 90%
  interval, because the 14 days of calibration don't fully capture month-to-month change.
- **October to December are hardest for every model.** Gradient boosting's error grows
  50–90% from July to December depending on horizon, likely because colder months bring
  more variable NO₂.
- Gradient boosting predicts in about **0.05 ms** per forecast in batch on Colab.

## The deployed forecaster

`scripts/deploy_forecaster.py` fits the same recipe once on all of 2024: for each
horizon, one gradient-boosting model on every target hour before 18 December, with
18 to 31 December setting the split-conformal 90% interval. It saves the four models
to `models/no2_gbm.joblib` (5 MB, committed; retraining takes about 20 s on a laptop)
and forecasts from each station's last reading in the newest EPA file:

- `results/forecasts.json` holds 1, 6, 12 and 24 h forecasts with 90% intervals and
  the EPA AQI category for 17 of the 20 stations. The newest file is
  `hourly_42602_2026.zip`, which runs to 31 March 2026 for most stations and to 30
  April for four. Three stations (Kansas City and St. Louis City, Missouri; Duchesne,
  Utah) have no 2026 data, so they get no forecast.
- These are real model forecasts but **not live**: they are issued from the last hour
  EPA has published, and every record says when. A live version needs a reading feed
  such as AirNow, which needs an API key.

The interval is one half-width per horizon for all stations (5.3, 11.0, 12.5 and 12.9
ppb at 1, 6, 12 and 24 h), set by a December calibration window. That makes it wide at
quiet rural stations and too narrow at the busiest one; see the next section.

## Out-of-time check: all of 2025

`scripts/monitor.py` scores the saved model on a file it never saw and compares it
with persistence on exactly the same hours. On all of 2025 (about 162,000 forecasts
per horizon, 20 stations), with the model trained on 2024 only:

| Horizon | Gradient boosting MAE | Persistence MAE | Improvement | 90% interval coverage | Worst station coverage |
|---|---|---|---|---|---|
| 1 h | 1.71 ppb | 1.87 ppb | 8% | 92.8% | 79.3% |
| 6 h | 3.44 ppb | 4.91 ppb | 30% | 93.9% | 80.7% |
| 12 h | 3.82 ppb | 5.74 ppb | 33% | 94.2% | 81.2% |
| 24 h | 4.01 ppb | 4.72 ppb | 15% | 94.0% | 79.9% |

The model keeps its backtest skill a full year later. Pooled, the intervals now
over-cover (93 to 94% against a nominal 90%) because December's calibration window is
noisier than most of 2025, while at the busiest Los Angeles station (06-037-4008-1)
only about 80% of outcomes fall inside them. Per-station intervals are the obvious fix
and are not done yet.

## Monitoring

`.github/workflows/ml-monitor.yml` runs every Monday, and on pull requests that touch
the model. It downloads the newest public EPA file, scores the committed model on
each station's last 28 days, and fails if, at any horizon:

- the model no longer beats persistence (skill below 0),
- 90% interval coverage falls below 80%, or
- MAE is more than twice the backtest MAE.

The thresholds were set before the first run. The job needs no secrets and finishes
in a few minutes. It installs the exact library versions the model was saved with
(`requirements-monitor.txt`), because the bundle is a pickle. A station whose mean
NO₂ moves by more than a factor of two is reported as a warning, not a failure.

On the last 28 days of the 2026 file (`results/monitor_latest.json`, run locally) the
model scored MAE 1.79 / 3.79 / 4.11 / 4.25 ppb at 1 / 6 / 12 / 24 h, 10 / 30 / 35 /
15% better than persistence, coverage 92 to 93%: passed. Two caveats. EPA updates these
files about twice a year, so most weekly runs re-check the same data. And a 28-day
window can flag an unusual month as well as a broken model.

## Forecast analyst (the agent)

`src/auraguard/agent` answers questions such as "Will NO₂ be high near Chicago
tomorrow morning?" or "How sure are you?". An LLM (`openai/gpt-oss-120b` on Groq,
temperature 0) plans and writes the answer, but it can only learn anything through
five typed tools over the committed artifacts:

| Tool | Returns |
|---|---|
| `list_stations(state?)` | station ids, county and state, location, whether a forecast exists |
| `get_forecast(station, horizon_h)` | point forecast, issue time, target time in UTC and local time, AQI category |
| `get_interval(station, horizon_h)` | 90% interval, with its backtest and 2025 coverage |
| `get_aqi_band(no2_ppb)` | EPA NO₂ AQI value and category (official 1-hour breakpoints) |
| `get_backtest_skill(horizon_h)` | backtest MAE, persistence MAE, improvement, coverage, the 2025 check |

Every draft then goes through a deterministic checker (`agent/checker.py`):

1. Every number must equal, or round from, a number in a tool output or in the
   question. A fraction may be written as a percentage. A number the model works out
   itself, such as a difference, is rejected on purpose.
2. Every station id must exist, and every US state or county it names must belong to
   a monitor or appear in the question.
3. A question about another pollutant, a place with no monitor within 80 km (checked
   against a small built-in gazetteer), or medical advice must get a plain refusal,
   and a medical question must get no health instruction.

A draft that fails gets the checker's feedback and one retry. If that fails too, the
answer is a template filled only from tool outputs, and the template is checked by the
same rules. The model never touches the forecasting model, the data or the network.

```bash
cp .env.example .env                    # then put your Groq key in it; .env is git-ignored
python -m auraguard.agent "Will NO2 be high near Chicago tomorrow morning?"
python -m auraguard.agent --json "How sure are you about the 6 hour forecast?"   # full trace
```

The backend exposes the same thing as `POST /api/v1/predictions/ask`.

## Agent eval

Run `python scripts/run_agent_eval.py` then `python scripts/agent_eval_report.py`.
The runner saves every raw record after each question, so a stopped run resumes where
it left off. Model `openai/gpt-oss-120b`, questions commit `af66b22`, code commit
`0eb4fdf`. Output: `results/agent_eval_raw.json`, `agent_eval_summary.json`,
`agent_eval.md`.

**Status: 19 of 40 questions run.** The Groq free tier caps this model at 200,000
tokens a day, and these 19 questions used the day's budget (about 10,000 tokens each).
Run so far: all 14 answerable and the first 5 of 8 ambiguous questions. Not run yet:
B06 to B08 (ambiguous), the 8 wrong-premise questions and the 10 out-of-scope
questions. Those last 18 are the refusal tests, so **no refusal rate is reported**.
Run the same two commands again after the quota resets (about 21 more questions) and
the report will cover all 40. Do not read the numbers below as the final result.

| Metric, 19 questions | First draft | Returned answer |
|---|---|---|
| Grounded: every number, station and place traceable to a tool output | 18/19 (95%) | 19/19 (100%) |
| Expected behaviour, string-matched | 17/19 (89%) | 17/19 (89%) |
| Refused although answerable | 1/19 (5%) | 1/19 (5%) |
| Tool calls correct (17 questions with expected calls) | 17/17 (100%) | |

| Category | Questions | Grounded before | Grounded after | Behaviour before | Behaviour after |
|---|---|---|---|---|---|
| answerable | 14 | 13/14 (93%) | 14/14 (100%) | 13/14 (93%) | 13/14 (93%) |
| ambiguous (B01 to B05) | 5 | 5/5 (100%) | 5/5 (100%) | 4/5 (80%) | 4/5 (80%) |

Of the 19 returned answers, 18 were the first draft and 1 was a retry after checker
feedback; none fell back to the template. There were no API errors. The checker
rejected one first draft (A11), which wrote the issue time with the wrong year, and
the retry corrected it. Mean tool calls per question: 1.7.

Latency per question: p50 2.0 s and p95 3.5 s of actual work. Including the time the
runner waited on Groq rate limits it was p50 34.8 s and p95 405.5 s, so the second pair
says more about the free tier than about the agent.

**Where it fails**

- **B05, Fargo.** The agent said there are no NO₂ monitors in North Dakota, but the
  station list has four, including Cass County (Fargo). A wrong claim with no number
  in it passes the checker, which only looks at numbers, station ids and places named.
- **A05, 60 ppb AQI band.** It gave the right category, "Moderate", but not the AQI
  value 57 that the question's string rules require.
- **Grounded does not mean right.** Grounding after the checker is 100% on these 19,
  but behaviour stays at 89%, because the checker cannot fix a refusal or an omission.
- **Small and string-graded.** 19 questions give wide error bars (one question is
  5 points), and grading is string matching, not human review.

## Reproduce

The backtest runs on a free Colab CPU runtime in about 10 minutes:
[`notebooks/backtest_colab.ipynb`](https://colab.research.google.com/github/CODER7657/AURA-GUARD/blob/master/ml/notebooks/backtest_colab.ipynb).
It downloads the EPA file onto the Colab machine only. Everything after the backtest
runs on a laptop in a few minutes. The EPA files are at
`https://aqs.epa.gov/aqsweb/airdata/hourly_42602_<year>.zip`.

```bash
python scripts/run_backtest.py --data hourly_42602_2024.zip --out results   # the backtest
python scripts/plot_results.py                                               # chart and table
python scripts/deploy_forecaster.py --train data/hourly_42602_2024.zip \
    --latest data/hourly_42602_2026.zip                                      # model and forecasts.json
python scripts/monitor.py --data data/hourly_42602_2025.zip --days 0 \
    --out results/monitor_2025.json                                          # out-of-time check
python scripts/export_app_metrics.py                                         # file the backend serves
python scripts/run_agent_eval.py && python scripts/agent_eval_report.py      # agent eval, needs a key
```

Tests use small synthetic series and a scripted stand-in for the LLM. They need no
download, key or network, and run in seconds:

```bash
pip install -e ".[dev]"
pytest
```

## Limitations

- **Not live.** Forecasts are issued from the last hour EPA has published (31 March
  or 30 April 2026), not from today's readings.
- **Intervals are not per station.** One half-width per horizon over-covers quiet
  stations and under-covers the busiest one (about 80% in 2025).
- **One training year, 20 monitors, 17 with forecasts.** Busy urban and quiet rural
  sites are mixed; results per station type are not reported separately.
- **No weather inputs.** Wind and boundary-layer height drive NO₂ and are the most
  likely source of further gains.
- **No satellite data yet.** NASA TEMPO NO₂ columns are the planned next feature.
- **The agent's checker is lexical.** It proves that numbers came from tools, not that
  the right number was used for the right station and hour. The eval's string-matched
  grading is what catches that, and and so far only for the 19 of the 40 questions that have been run.
