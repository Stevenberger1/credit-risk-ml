3️⃣ The correct general steps (start → finish)

Here is the exact, professional-grade workflow we will build and test using Logistic Regression only.

Phase A — Infrastructure (what we build now)

Step 1. Define evaluation metrics (one place, reusable)
→ evaluation/metrics.py

Step 2. Define a generic model evaluator
→ evaluation/evaluator.py
This evaluates any trained model in exactly the same way.

Step 3. Define result storage & comparison format
→ evaluation/results.py

Step 4. Define a baseline experiment runner
→ experiments/run_baselines.py

At the end of Phase A:

You can evaluate any model

You can compare results

You can trust the pipeline

Phase B — Validation (still only Logistic Regression)

Step 5. Run Logistic Regression through the full workflow

load data

build preprocessor

build model

fit

evaluate

save results

If anything breaks → we fix the framework, not the model.

Phase C — Expansion (later, not now)

Only after Phase B works perfectly:

add Random Forest

add XGBoost

add LightGBM

reuse the exact same evaluator