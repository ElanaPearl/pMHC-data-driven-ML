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
print("Loaded embeddings.")
## TODO: What's the task? What do we do with these scores?

metric_info = {"CWEntropy": {"metric_object": methods.LabelQuality(method="confidence_weighted_entropy")},
               "SelfConfidence": {"metric_object": methods.LabelQuality(method="self_confidence")},
               "NormalizedMargin": {"metric_object": methods.LabelQuality(method="normalized_margin")},
               }

# Compute risk-coverage tradeoffs for each metric
for metric_name, metric_dict in metric_info.items():
    print(f"Running {metric_name}")
    metric_object = metric_dict["metric_object"]
    metric_object.fit(embeddings=val_embeddings, predictions=val_predictions, labels=val_labels)
    
    val_scores = metric_object.get_score(embeddings=val_embeddings, predictions=val_predictions, logits=val_logits, labels=val_labels)

## TODO: What's the task? What do we do with these scores? 
