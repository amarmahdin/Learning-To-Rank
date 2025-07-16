import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

class RankingEvaluator:
    def __init__(self, k=None):
        self.k = k

    def compute_group_ndcg(self, df, group_column='assigned_technology', label_col='label', pred_col='pred'):
        ndcg_scores = []
        for group_value, group in df.groupby(group_column):
            if group[label_col].nunique() < 2:
                continue
            y_true = np.asarray(group[label_col]).reshape(1, -1)
            y_pred = np.asarray(group[pred_col]).reshape(1, -1)
            if self.k:
                score = ndcg_score(y_true, y_pred, k=self.k)
            else:
                score = ndcg_score(y_true, y_pred)
            ndcg_scores.append(score)
        if not ndcg_scores:
            return 0.0
        return np.mean(ndcg_scores)

    # Tambahan: evaluasi ranking pada instance (bukan pairwise!)
    def evaluate_instance_ranking(self, df, label_col='delay_norm_global', pred_col='score', group_col='assigned_technology'):
        techs = df[group_col].unique()
        ndcgs = {}
        for tech in techs:
            tech_df = df[df[group_col]==tech]
            if len(tech_df) < 2:
                ndcgs[tech] = np.nan
                continue
            y_true = np.asarray(tech_df[label_col]).reshape(1, -1)
            y_pred = np.asarray(tech_df[pred_col]).reshape(1, -1)
            score = ndcg_score(y_true, y_pred, k=self.k) if self.k else ndcg_score(y_true, y_pred)
            ndcgs[tech] = score
        mean_ndcg = np.mean([v for v in ndcgs.values() if not np.isnan(v)])
        return ndcgs, mean_ndcg