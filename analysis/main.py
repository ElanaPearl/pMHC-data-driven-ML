## Run simple risk-coverage tradeoffs 
import numpy as np
from data import get_classification_data
import methods
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

test_embeddings, test_logits, test_predictions, test_labels = get_classification_data()

# Gotta do some arbitrary splits here
val_embeddings, test_embeddings, val_logits, test_logits, val_predictions, test_predictions, val_labels, test_labels = train_test_split(test_embeddings, test_logits, test_predictions, test_labels, test_size=0.5, random_state=42)

# Standard evaluation for Selective Classification settings as a warmup
# https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
def risk_coverage_curve(scores, predictions, labels, num_bins=10):
    quantiles = np.linspace(1, 0, num_bins+1)[1:]
    risks = []
    coverages = []
    for q in quantiles:
        sq = np.quantile(scores, q)
        coverage = (scores >= sq).mean()
        risk = (predictions[scores >= sq] != labels[scores >= sq]).mean()
        risks.append(risk)
        coverages.append(coverage)
    return {"risk": risks, "coverage": coverages}


metric_info = {"Entropy": {"metric_object": methods.Entropy()},
               "TrustScore": {"metric_object": methods.TrustScore()},}
               #"KNNConfidence": {"metric_object": KNNConfidence()}

# Compute risk-coverage tradeoffs for each metric
for metric_name, metric_dict in metric_info.items():
    print(f"Running {metric_name}")
    metric_object = metric_dict["metric_object"]
    metric_object.fit(embeddings=val_embeddings, predictions=val_predictions, labels=val_labels)
    
    test_scores = metric_object.get_score(embeddings=test_embeddings, predictions=test_predictions, logits=test_logits)
    metric_dict.update(risk_coverage_curve(test_scores, test_predictions, test_labels))

    
sns.set_context("poster")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for metric_name, metric_dict in metric_info.items():
    ax.plot(metric_dict["coverage"], metric_dict["risk"], label=metric_name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel(r"$\mathcal{R}(\hat{y}, y)$")

ax.set_title("Selective Classification Risk-Coverage Tradeoffs")
fig.tight_layout()
fig.savefig("risk_coverage_tradeoffs.png")