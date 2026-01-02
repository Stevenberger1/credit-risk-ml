from sklearn.pipeline import Pipeline # The Pipeline class from sklearn.pipeline is used to chain together multiple steps of a machine learning workflow into a single, cohesive object
from sklearn.linear_model import LogisticRegression 


def build_logistic_regression_pipeline(preprocessor):
    '''This function answers one question:
    “Given a preprocessing definition, how do I build my baseline model?”
    '''


    logistic_regression = LogisticRegression( #This creates the classifier part of the pipeline.
        solver="liblinear", # Chooses the optimization algorithm. very stable for small/medium datasets , works well with binary classification.
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline( #This creates the full pipeline by combining the preprocessor and the classifier.
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", logistic_regression),
        ]
    )

    return pipeline

'''
We will tune after:

evaluating precision/recall

analyzing ROC & PR curves

understanding tradeoffs
'''

