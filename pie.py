import pandas as pd
from pypastry.experiment import Experiment
from pypastry.predictors.document import DocumentClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold


def get_experiment():
    dataset = pd.read_csv('test_dataset.csv')
    # categories = [
    #     'alt.atheism',
    #     'talk.religion.misc',
    # ]
    # dataset = fetch_20newsgroups(subset='train', categories=categories)
    # label_column = dataset.columns[1]
    predictor = DocumentClassifier()
    cross_validator = StratifiedKFold(n_splits=2)
    scorer = make_scorer(f1_score)
    label_column = 'relevant'
    print(dataset)
    return Experiment(dataset, label_column, predictor, cross_validator, scorer)
