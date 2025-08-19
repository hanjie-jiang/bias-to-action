---
title: Week1P4 - ML Fundamentals
---

### Resources
[[ç™¾é¢æœºå™¨å­¦ä¹ ç®—æ³•å·¥ç¨‹å¸ˆå¸¦ä½ åŽ»é¢è¯•.pdf]]

### Feature Engineering

Two types of data:
- Structured / Tabular data: Could be viewed as a data table from the relational database, which every columns has their clear definition, including **numerical** and **categorial** data types.
- Unstructured data: Includes **text, image, audio, video data**, and the information that this type of data contains cannot be represented easily as a numerical value, and also they do not have clear categorical definition, furthermore, the size of these data are not identical.

##### Normalization of Features

###### Why does one need to do normalization on numerical features?

In order to eliminate the magnitude impact between features, we should always do normalization to the features that we use, i.e. to uniformly normalize all the features to a similar range, so that it could **help compare between different metrics**. There are two different types of normalization that people most commonly use:
- *min-max scaling*: It linearly changes the original data so that the data could be projected to [0, 1] range so that it is an equal ratio transformation of the original data:
$$X_{\text{norm}} = \frac{X-X_{\text{min}}}{X_{\text{max}-X_{\text{min}}}}$$
- *Z-Score normalization*: It would project the original data to a mean of 0 and variance = 1 distribution. Specifically, assume that the original feature has mean $\mu$ and variance $\sigma$ , then the normalization equation would be defined as:
$$Z = \frac{x-\mu}{\sigma}$$ 
Using stochastic gradient descent (SGD) as an example, when two numerical features, $x_1$ of range [0,10] and $x_2$ of range [0,3], then when the $x_1$ and $x_2$ are not normalized, the ![[Screenshot 2025-08-05 at 8.22.11 PM.png]] gradient descent would not be as efficient as when one does the normalization of the features. However, feature normalization is not always working. In real life, <span style="background-color: #FEE9E7"> whenever a model utilizes SGD, it is suggested to use the normalization, including linear regression, logistic regression, support vector machine, neural networks, whereas decision tress it does not help. </span> As for decision tree models, the node split usually is determined by the data and how much [[^1]information gain ratio](https://en.wikipedia.org/wiki/Information_gain_ratio) that data contains about X. This information gain ratio is not impacted by whether the feature has been normalized, rather it would not change the information gain of the specific feature X.

[^1]: need to work on the definition of this and learn more about information theory

##### Categorical Features
Categorical features include male / female, blood type (A,B,AB,O) and etc, which can only select values from a finite set of choices. Categorical features original input are mostly strings. Despite that **decision trees and some other numbers of models can directly take in the strings, for logistic regression or SVM models, the categorical features need to be translated to numerical form** so that they could properly work.

###### How to do feature engineering on categorical features?

One would need to encode the features to a higher dimensional vector to represent them in the model.
- **ordinal encoding**: usually used to treat those data that has ordinal sequence, for example when scoring we have high > middle > low, then the ordinal encoder would help to describe this type of sequence via giving it a numerical ID. For example, we could represent high as 3, middle as 2 and low as 1 in this case, which helps retain the high to low relationship.
- **one-hot encoding**: usually used to treat features that do not have ordinal relationships, for example, for blood type, one could directly use the [1,0,0,0], [0,1,0,0], [0,0,1,0] and [0,0,0,1] to represent the different types. Note:
	- use of sparse vector for saving space
	- high-dimensional features can be difficult in following scenarios: 1) K-nearest neighbors, the distance between two high-dimensional vectors can be hard to measure, 2) logistic regression, the parameters can increase with higher dimensions, thus causing overfitting problems and 3) only some of the dimensions could be helpful when doing clustering or predictions, so one could think to reduce dimensions with feature selections.
- **binary encoding**: using binary to do a hash mapping on the original category ID, this can help save space when comparing with the one-hot encoding as it is usually of fewer dimensions.

##### High Dimensional Feature Crosses

###### What are feature crosses? And how to deal with high-dimensional feature crosses?

Using single features to combine them together via dot-product or inner-product, one can get a combination of two features to help represent nonlinear relationships.

Using logistic regression as an example, when a data set contains feature vector $X=(x_1, x_2, ..., x_k)$ then one would have $Y = \text{sigmoid}(\sum_i \sum_j w_{ij} <x_i, x_j>)$ . $w_{ij}$ is of dimension $n_{x_i}\cdot n_{x_j}$ . But when $n_{x_i} \times n_{x_j}$ is huge, especially in use cases of website customers and number of goods, this can be really huge dimension. So **one way to get around this is to use a k-dimensional low-dimension vector (k << m, k << n). Now,  $w_{ij} = x_i' \cdot x_j'$ and now the number of parameters one needs to tune is $m\times k + n \times k$ . This can also be viewed as the [[^2]matrix vectorization](https://lumingdong.cn/recommendation-algorithm-based-on-matrix-decomposition.html), that has been widely used in the recommendation systems. **

[^2]: Please read through the recommendation system based on matrix vectorization to get a better idea on how recommenders are built based on SVD and matrices

We have understood how to use dimension reduction to reduce the number of parameters that the model needs to learn given a feature cross of two high-dimensional features. <span style="background-color: #FEE9E7"> But in reality, we are facing a variety of high-dimensional features. So a single feature crosses of all the different pairs would induce 1) too many parameters and 2) overfitting issues. </span>
###### How to effectively select the feature combinations?
We introduce a feature cross selection based on decision tree models. Taking CTR prediction as an example, assume that the input includes age, gender, user type (free vs paid), searched item type (skincare vs foods), etc. We could thus make a decision tree from the original input and their labels. ![[Screenshot 2025-08-05 at 9.30.27 PM.png]] We could then view the feature crosses from the tree, that contains four different type of pairs:
1. age + gender
2. age + searched item type
3. paid user + search item type
4. paid user + age
How to best construct the decision trees? One can use the [Gradient Boosting Decision Treeï¼ŒGBDT](https://medium.com/@ruchi.awasthi63/gradient-boosted-decision-tree-clearly-explained-bd1d8c7d9923) or use[ the link](https://neptune.ai/blog/gradient-boosted-decision-trees-guide) to get a better idea of the algorithm. The idea behind is that whenever before constructing a decision tree, we first calculate the error from the true value and iteratively construct the tree from the error.

##### Textual Descriptive Models
Related Content: [[Week3_Ngram_Language_Modeling]]

Text is a category of unstructured data. How to work with textual data has always been one of the most important research directions.

###### What are some of the textual descriptive models what pros and cons each have?
- Bag of words: Consider each article as a bag of words, ignoring the sequence of how each word appears. Specifically, it separates the entire paragraph of texts at word unit and represent each paragraph as a long vector. Each dimension in the vector is a word, and the weight represents how important the word is in the original article. 
- TF-IDF (Term Frequency-Inverse Document Frequency): Is often used to calculate the weight of the words, $\text{TF-IDF}(t,d)=\text{TF}(t,d) \times \text{IDF}(t)$ , where $\text{TF}(t,d)$ represents the frequency of word t in document d, whereas $\text{IDF}(t)$ is the reverse document frequency to measure word t's importance in grammar, corresponds to equation $$\text{IDF}(t) = log^{\frac{\text{total article}}{\text{total article that contains word} t +1}}$$ the general meaning behind is that if a word appears so in various articles, then it means that it is a commonly used word, hence it would not contribute much in differentiating the specific meaning behind each articles, hence it should be penalized when weighting.
- N-gram: when "natural language processing" being separated into 3 words as word unit, the meaning of this phrase is totally different from it is now, hence usually we could add n words as a feature unit into the vector to form the N-gram model. 
- Topic Model
- Word Embedding: word embedding is a family of word vector models, the main idea is to project each word to a low-dimensional space (K = 50 -300 dimensions) using a dense vector. Each dimension in K-dimension would be viewed as a implicit topic. 
<span style="background-color: #FEE9E7">In general, in shallow learning models (traditional ML models), a good feature engineering step can help extremely good performance. Deep learning on the other hand, could help us with an automated feature engineering way via hidden layers. Hence, it makes sense for the deep learning model to beat the shallow learning model in general. Recurrent neural network and convolutional neural network are both good at capture the characteristics of the text while lowering the number of parameters that the model needs to learn, which can expedite the speed of training and also lower the risk of overfitting. </span>

##### Word2Vec
One of the most common word embedding models, it is actually a shallow neural network. It can be of two different types of structures: 
1. Continuous Bag of Words
2. Skip-gram

###### How does word2vec work? what is the difference between word2vec and LDA (Latent Dirichlet allocation)
- Continuous Bag of Words
	- Goal is to use contextual words that predict the probability of the current word to appear.
	- Structure: 
		- input layer: w(t-2), w(t-1), ..., w(t+1), w(t+2) using one-hot encoding
		- projection/hidden layer: sum(probability)
		- output layer: w(t) using softmax
- Skip-gram
	- Goal is to use the current word to predict the probability of each contextual word.
	- Structure:
		- input layer: w(t) using one-hot encoding
		- projection/hidden layer
		- output layer: w(t-2), w(t-1), ..., w(t+1), w(t+2) using softmax

##### Image Data not sufficient - Cold Start
When doing machine learning modeling, one very big problem that everyone may face would be not sufficient training data. 

###### What would you do if the training data is not sufficient? How to mitigate this issue?
#transfer-learning #generative-adversial-network #image-processing #over-sampling 
Information that a model can provide include 1) information from training and 2) heuristic information that people provide from model formation (including design / learn / deduct). When training data not enough, it means that the model lacks information from training data, but need more a priori. a priori can be effective on models, including certain internal structure of the model, assumption or constraints. a priori can also be applied to datasets, for example using certain assumption to change / tune or expand the training data so it contains more effective information, which can facilitate model training and learning.

##### Overfitting
#overfitting
One big problem that comes **from not enough data is overfitting**, which is that the model performs well on training set but the evaluation / prediction set is not good. The treatment can come from two different categories:
- methods based on models that decrease the risk of overfitting
	- simplify model - downgrade from non-linear to linear model
	- apply constraints to shrink hypothesis space - L1 / L2 regularization
	- integrated training
	- dropout hyperparameters
- data augmentation: manipulating data to expand the data set
	- image space manipulation
		- rotation / shrinkage / expansion / crop of the original image when working with image data
		- addition of noise to the image
		- color change of image
		- hue / contract / brightness of image
	- image feature engineering / extraction
		- data expansion or #over-sampling via SMOTE (Synthetic Minority Over-sampling Technique)
		- using GAN or other generative methods for good samples
	- transfer learning from other models and data
		- using pre-trained general model from big dataset, we could fine-tune specifically using the small datasets


### Model Evaluation
#### Evaluation metrics and their limitations
When doing model evaluation, the classification / sort / regression problems seems to always use different metrics for evaluation. 
##### Accuracy and its limitations
The accuracy only measures the number of correct labels divided by the number of total labels. This can potentially lead to a issue **when the number of labels are limited in the dataset**. When negative samples composed 99% of the data, if every label is a negative one, we still get 99% accuracy. So, if we use more effective mean accuracy that quantifies the mean accuracy under each category, it would be a better metrics to work with.
##### Precision & Recall and their balance
###### Concept of Precision & Recall
Now we need to introduce the concept of precision and recall. #precision
#recall #F1score
Precision cares about the correctness of positive predictions, whereas recall cares about coverage of actual positives.  <span style="background-color: #FEE9E7">Precision and recall trade off via the decision threshold.</span> In a binary classification problem:
$$\text{Precision} = \frac{N_{\text{true positive}}}{N_{\text{true positive}} + N_{\text{false positive}}} = \frac{N_{\text{true positive}}}{N_{\text{positive predictions}}} $$

$$\text{Recall} = \frac{N_{\text{true positive}}}{N_{\text{true positive}} + N_{\text{false negative}}} = \frac{N_{\text{true positive}}}{N_{\text{actual positives}}} $$
The F1 score is their harmonic mean:
$$\text{F1} = \frac{2(\text{Precision})(\text{Recall})}{\text{Precision} + \text{Recall}} = \frac{2N_{\text{true positive}}}{2N_{\text{true positive}}+N_{\text{false positive}}+N_{\text{false negative}}} $$
this value ranges from 0 to 1 and penalizes imbalance, thus when either precision or recall is low, F1 drops sharply. <span style="background-color: #FEE9E7"> F1 should be used when false positives and false negatives matter about equally, especially with imbalanced classes. </span>
###### Precision & Recall in Ranking / retrieval variants
```
def precision_at_k(ground_truth_set, ranked_list, k):
    return len(set(ranked_list[:k]) & ground_truth_set) / k
```

```
# when there are more than one query / user / example that we would like to test on our predictions, we use the weighted average of the precision_at_k.
def mean_precision_at_k(ground_truth_sets, ranked_lists, k):
    # ground_truth_sets and ranked_lists are aligned lists
    return sum(precision_at_k(g, r, k) for g, r in zip(ground_truth_sets, ranked_lists)) / len(ground_truth_sets)
```
- **Precision@k** â†’ for **one** case $q$ (one list).
- **Mean Precision@k** â†’ average of those values over **all** cases $q \in Q$.

**Example**: when dealing with video vague search functionality, it seems that the search ranking model can return the top 5 precision pretty high, however, the user in reality still cannot find the videos they want, especially those unpopular ones. Where does this problem coming from?

**Root cause analysis**: Coming back to the example above, the top 5 precision being really high, meaning that the model can get the true positive results on a pretty good level with a certain set of positive predictions; however, when it comes down to cases where users would like to find not so popular videos, the precision of ranks can be rather no so useful as the user is looking for not so well-defined labels, hence the good precision of popular videos would not be helpful for this case as <span style="background-color: #FEE9E7">model is not providing all the relevant videos to the user and this is a problem of not so good recall rate. </span> Let's say for the top 5 results, the precision@5 to be 100%, meaning that the correctness of the positive results is pretty higher, however, the recall@5 can still be 5%, meaning that only predicted 5 true positives although there are 100 actual positives involved. When doing model evaluation, it means that we should be focusing on both precision and recall, and also using different top N values for observations. 

Hence, in general, when people evaluate the goodness of a sort algorithm, they also look at the P-R curve, where in this curve, the x-axis corresponds to recall rate whereas the y-axis corresponds to precision rate. 

###### Use of P-R Curve for model evaluation and threshold choice
![[p-r_curve.png]]
Each data point on the curve corresponds to a precision-recall combination at a certain threshold for True samples of choice, for example 0.95 / 0.9, etc. The closer to the origin (0,0) point, the bigger the threshold is.

###### How to pick the threshold in practice
- **Capacity-constrained:** If reviewers can handle 300 cases/day, pick the smallest threshold that yields â‰ˆ300 flags/day; report the resulting (Precision, Recall).
- **Recall target:** If policy demands **â‰¥95% recall**, choose the lowest threshold achieving that, then report precision (and expected review load).
- **Cost-based:** Minimize $\text{Cost}_{\text{false positives}}\cdot{\text{False Positives}}+\text{Cost}_{\text{false negatives}}\cdot{\text{False Negatives}}$ over thresholds.
Also report **AUPRC** to compare models independent of a single threshold (higher is better, especially with class imbalance).
##### Root-mean Squared Errors (RMSE)
###### Definition of Root-mean squared error
#root-mean-squared-error #rmse 

$$ RMSE = \sqrt{\frac{\sum_{i=1}^{n}{(y_i - \hat y_i)^2}}{n}} $$

Root-mean squared error has long been used as the metric for evaluating the regression model.

**Example**: as a streaming company, one would say that prediction of traffic for each series can be really important when it comes down to ads bidding and user expansion. One would like to use a regression model to predict the traffic trend of a certain series, but whatever regression model that one uses, the RMSE metric ends up being really high. But, in reality, the model 95% of the time predict error is less than 1%, with really good prediction results. What might be the reason of this extraordinarily good results?

**Root cause analysis**: From what the example, says there are two possible ways for the RMSE to be ineffective: 1) n being really small hence at this moment, the calculated error cannot be measurable anymore, 2) all the errors between actual value and predicted value are over- / under-predicting that the summation at the end being really high, however, in reality it is not the case and <span style="color: #FF6961">3) one outlier being really off when comparing with other data points, it is contaminating the RMSE to be really big. </span> Coming back to the question, as 95% of the time to model has really good prediction error hence it means the other 5% of the time the model can be really off with big outliers and it could happen when a series with small traffic / newly come-out / newly accoladed could produce this big error.

**How to solve:** 1) When we think these outliers are noises, then we need to filter them out at the early stage when doing data cleaning, 2) If we do not think they are noises, then we need to further improve the prediction capability of our algorithm so that we could somehow model the formation of these outliers. and 3) We could also use a better metric for the model evaluation. There are indeed better evaluation metrics that are of better robustness than RMSE, for example, Mean Absolute Percentage Error (MAPE):

###### Definition of Mean Absolute Percentage Error
#mean-absolute-percentage-error #mape 

$$MAPE = \sum_{i=1}^n{|\frac{(y_i - \hat y_i)}{y_i}|\cdot\frac{100}{n}}$$

When comparing with RMSE, MAPE normalizes the error rate of each data point to mitigate the outlier impact from the absolute error.

##### Expanding on the regression evaluation metrics
#smape #regression-metrics
###### Quick definitions

Let $y$ be the true value and $\hat y$â€‹ the prediction.    
**sMAPE** (common form): 
$$\frac{100}{n}\sum\frac{2|y-\hat y|}{|y|+|\hat y|}$$
###### When to use which
- **Use RMSE** when:
    - Big errors are much worse than small ones (squared penalty).
    - The target never hits zero/near-zero and units are meaningful (e.g., dollars, Â°C).
    - You care about _calibration_ and smooth optimization (differentiable).
- **Use MAPE** when:
    - Stakeholders want an average **percentage** error thatâ€™s easy to read.
    - True values are **strictly positive and not near zero** (e.g., revenue, demand > 0).
    - Youâ€™re okay that over-forecasts and under-forecasts are weighted differently (MAPE tends to penalize under-forecasting less when $y$ is small).
- **Use sMAPE** when:
    - You want a percentage-like metric thatâ€™s **less explosive near zero** than MAPE.
    - You have occasional zeros or tiny values.
    - You accept that sMAPE has its own quirks (bounded but not perfectly â€œsymmetricâ€ in practice).
###### Strengths & gotchas (TL;DR)
- **RMSE**
    - âœ… Sensitive to large mistakes (good if that matches cost).
    - âš ï¸ Outlier-heavy data can dominate the score.
    - âš ï¸ Scale-dependentâ€”hard to compare across series with different scales.
- **MAPE**
    - âœ… Intuitive (%).
    - âš ï¸ Undefined at y=0; huge when y â‰ˆ 0.
    - âš ï¸ Can favor **under-forecasting** for small y.
- **sMAPE**
    - âœ… Handles zeros better; bounded.        
    - âš ï¸ Still quirky near zero and not a true â€œcostâ€ for optimization.
    - âš ï¸ Different papers/tools use slightly different variantsâ€”state your formula.
###### Other basic metrics you should know
- **MAE**: Robust to outliers vs RMSE; easy to explain (units).
###### Simple decision guide
1. **Zeros or tiny targets?**
    - Avoid plain MAPE. Prefer **sMAPE**
2. **Large errors are very costly?**
    - Use **RMSE** (or set a business-weighted loss).
3. **Need % interpretability across series?**
    - Use **sMAPE**, or **MASE** (if comparing to a baseline).
4. **Care about relative ratios (under/over by Ã—%)?**
    - Use **RMSLE/MSLE** (with positive targets).
5. **Mixed scales or many series?**
    - **WAPE** or **MASE** are safe, comparable choices.
###### Practical tips
- If you must report a % and have zeros, say: _â€œWe use sMAPE (formula shown) instead of MAPE to handle zeros; we also report WAPE for scale-free comparability.â€_
- Always **state the exact formula** you use (especially for sMAPE) to avoid confusion.
- <span style="color:#FF6961">Consider reporting **two metrics**: one business-facing (% like WAPE/sMAPE) + one technical (MAE/RMSE).</span>

Overall, one should always report a pair / set of MECE metrics to evaluate their algorithms to better understand & discover the problems in the model, to better solve cases in real business settings.

##### ROC Curves
#roc-curve #auc
Binary classifiers are the mostly used and applied classifier in the ML industry. There are a lot of different metrics that one could use for evaluate the binary classifiers, including precision, recall, F1 score and P-R curve. But these metrics are only reflecting one aspect of the model. Hence, ROC curves can be of really good use. 

###### What is a ROC curve
ROC curves are called receiver Operating Characteristic Curves, which established from the military field and are often used in the medical industry as well. This curve's x-axis is the false positive rate, whereas the y-axis is the true-positive rate. 

$$\text{False Positive Rate} = \frac{\text{False Positive}}{\text{Negative}}$$
$$\text{True Positive Rate} = \frac{\text{True Positive}}{\text{Positive}}$$
**Example**: There are 10 patients, where in there are 3 positive cancer patients, and the rest are negative patients. The hospital decides to do diagnosis on these customers and figured that 2 are true positive cancer patients. In this case:

$$\text{False Positive Rate} = \frac{\text{False Positive}}{\text{Negative}} = \frac{1}{7}$$
$$\text{True Positive Rate} = \frac{\text{True Positive}}{\text{Positive}}=\frac{2}{3}$$
###### How to draw a ROC curve
- What is needed
	- True labels $y \in \{0,1\}$
	- A **score** for the positive class per item (probability or decision score).  

| Sample Number | True Label | Model Output Probability as Positive |
| ------------- | ---------- | ------------------------------------ |
| 1             | Positive   | 0.9                                  |
| 2             | Positive   | 0.8                                  |
| 3             | Negative   | 0.7                                  |

From this example, we could then plot out the true positive rate (TPR) as the x-axis and false positive rate (FPR) as the y-axis for the curve, hence getting the ROC curve. There is a more direct way to plot the ROC curve as well:

- Getting the number of Positive & Negative samples, i.e. assuming number of positive samples to be P and negative to be N.
- Getting the x-axis labels to be the count of negative samples, and y-axis labels to be the count of positive samples, then use the model output probability to do sorting of the samples
- Now draw the ROC curve from origin, whenever seeing a positive sample to draw a vertical line segment of +1 increment on y-axis, whenever seeing a negative sample then we draw a horizontal line segment along the x-axis until we reach the final sample with curve ending at (1,1).

```
import numpy as np
import matplotlib.pyplot as plt

def roc_curve_from_scores(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    order = np.argsort(-y_score)        # sort by score desc
    y_true = y_true[order]

    P = y_true.sum()
    N = len(y_true) - P
    # cumulative TP/FP as we lower the threshold
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    # rates; guard against division by zero
    tpr = tps / (P if P > 0 else 1)
    fpr = fps / (N if N > 0 else 1)

    # include the origin (0,0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return fpr, tpr

def auc_trapezoid(x, y):
    return np.trapz(y, x)

# --- Example usage (replace y_true/y_score with your data) ---
rng = np.random.default_rng(0)
y_true  = rng.integers(0, 2, size=1000)
# make scores vaguely correlated with y_true
y_score = 0.2*rng.random(1000) + 0.6*y_true + 0.2*rng.random(1000)

fpr, tpr = roc_curve_from_scores(y_true, y_score)
roc_auc = auc_trapezoid(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

```

###### How to calculate the AUC (area under curve)?
As simple as it could be, AUC is the area under the ROC curve, which can quantitatively reflect the model performance based on ROC curve. It is simple to calculate AUC along RUC x-axis. Due to that ROC curve tends to be above y=x, AUC values are usually between 0.5-1. The bigger the AUC is, the better the classifier is as the more likely that the classifier put the true positive samples at the front. 

```
# Re-render the ROC curve example and save a downloadable image.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Example ROC points (FPR on x-axis, TPR on y-axis), sorted by FPR
fpr = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
tpr = np.array([0.0, 0.4, 0.7, 0.85, 1.0])

# AUC via trapezoidal rule
auc_value = np.trapz(tpr, fpr)

# Compute per-segment trapezoid areas
x_left = fpr[:-1]
x_right = fpr[1:]
y_left = tpr[:-1]
y_right = tpr[1:]
width = x_right - x_left
avg_height = (y_left + y_right) / 2.0
area = width * avg_height

seg_df = pd.DataFrame({
    "x_left (FPR_i)": x_left,
    "x_right (FPR_{i+1})": x_right,
    "y_left (TPR_i)": y_left,
    "y_right (TPR_{i+1})": y_right,
    "width = Î”x": width,
    "avg_height = (y_i + y_{i+1})/2": avg_height,
    "trapezoid_area = width * avg_height": area
}).round(4)

# Plot the ROC curve and shade trapezoids
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, marker='o', label=f"ROC curve (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
for i in range(len(fpr) - 1):
    x_pair = [fpr[i], fpr[i+1]]
    y_pair = [tpr[i], tpr[i+1]]
    plt.fill_between(x_pair, [0, 0], y_pair, alpha=0.2)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("ROC Curve with Trapezoids (AUC via Trapezoidal Rule)")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":")
plt.tight_layout()

# Save a copy to disk so the user can download it
plot_path = "/mnt/data/roc_trapezoids.png"
plt.savefig(plot_path, dpi=160)
plt.show()

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Per-segment trapezoid areas (sum equals AUC)", seg_df)

print(f"AUC (trapezoidal) = {auc_value:.4f}")
print(f"Sum of trapezoid areas = {area.sum():.4f}")
```

We have touched on the P-R curve for evaluating classification or sort algorithms. Comparing with P-R curve, there is one important character of ROC curve, which is that when positive / negative sample distribution change significant, the ROC curve shape could stay rather consistently whereas the P-R curve shape would be changing. This makes the ROC curve to mitigate the interference from diverse test sets and could more objectively evaluate the algorithm. In reality, when positive counts are much less than the negative counts, when switching dataset the data can be of big change, so a stable and robust evaluation would be important. Hence, usually ROC can be used in more variety of scenarios and could be utilized in sort / recommendation / ads. 

##### What each curve shows
- **ROC**: y = True Positive Rate (recall), x = False Positive Rate.  
    _â€œHow well do I separate positives from negatives overall?â€_    
    _"If I take the items my model flags as positive, how many are actually positive?â€
- **PR**: y = Precision, x = Recall.  
    _â€œWhen I go after positives, how clean are my catches?â€_
    _â€œAs I move the threshold, how well do I trade off catching positives vs accidentally flagging negatives?â€_
##### When to use which
- **Use PR (Precisionâ€“Recall) when positives are rare or review capacity is limited.**  
    Examples: fraud (â‰¤1%), disease screening, anomaly detection, search/retrieval, human-in-the-loop queues.  
    Why: PR focuses on the _quality of retrieved positives_. Baseline matters: random **AUPRC â‰ˆ prevalence** (e.g., 1% positives â†’ random AUPRC = 0.01).
- **Use ROC when classes are roughly balanced or you care about both error types evenly.**  
    Examples: many general classifiers, spam vs ham with moderate prevalence, A/B classifiers in balanced datasets.  
    Why: ROC is insensitive to class imbalance and summarizes ranking quality across thresholds. Random **AUC-ROC = 0.5**.
##### Intuition about imbalance
- With 1,000,000 negatives and 1,000 positives, an FPR of **0.5%** looks tiny on ROC, but itâ€™s **5,000 false alarms**â€”precision will be poor.  
    PR makes this visible; ROC can look deceptively â€œgreat.â€
##### How to choose in practice
- **Rare positives or ops-constrained?** Prefer **PR** (and report Precision/Recall at your operating threshold or **Precision@k**).
- **Balanced costs/distribution?** **ROC** is fine (and stable).
- **Comparing models broadly?** Report **both** AUC-ROC and AUPRC, plus a point metric at your intended threshold.
##### Reading the curves
- **ROC**: closer to top-left is better; AUC near 1 is strong.
- **PR**: higher curve is better; sustaining high precision as recall grows is ideal.
- Curves can **cross**. Pick the model thatâ€™s better in the **recall region you care about** (e.g., recall â‰¥ 0.9). Consider **partial AUC** (ROC) or **AUPRC over a recall range**.
##### What to report (good default)
- **AUPRC + AUC-ROC** (global picture)
- **(Precision, Recall)** (or $F_\beta$) at the **chosen threshold**
- If capacity-limited: **Precision@k** (and expected volume flagged)

##### Use of cosine distance
#cosine-similarity #cosine-distance #euclidean-distance
How to evaluate the distance between samples can also define the optimization target and training method. In ML problems, we usually take the features to be of vector form, so when analyzing the two feature vector similarity, we could use cosine similarity. The cosine similarity can range from -1 to 1, where when two vectors are exactly the same, the cosine similarity becomes 1. Hence, when looking at distances, 1-cosine similarity becomes the cosine distance. Overall, the cosine distance is [0,2] and the same two vectors their cosine distance becomes 0.

###### Definition of Euclidean Distance & Cosine Distance

**Euclidean Distance**
For vectors $x,y\in\mathbb{R}^d$:

$$d_{\text{Euc}}(x,y)=\sqrt{\sum_{i=1}^{d}(x_i-y_i)^2} \in [0,\infty) $$

- **What it measures:** straight-line (L2) distance in space.
- **Sensitive to scale/magnitude:** doubling a vector doubles distances.
- **Squared form:** sometimes use $\|x-y\|^2$ (no square root) for speed/convexity.~

**Cosine Distance**
Start with cosine **similarity**:

$$\text{cos\_sim}(x,y)=\frac{x\cdot y}{\|x\|\,\|y\|}\in[-1,1]$$

Cosine **distance** (common definition):
$$d_{\text{cos}}(x,y)=1-\text{cos\_sim}(x,y)\in[0,2]$$

- **What it measures:** difference in **direction** (angle) only.
- **Scale-invariant:** multiplying a vector by a positive constant doesnâ€™t change it.

Overall, on unit vectors, Euclidean and cosine distances are monotonic transforms.
Also, on a unit circle, one would see:
$$\|A-B\|=\sqrt{2(1-cos(A,B))}$$
- **When to use which**
    - Use **Euclidean** when magnitude matters (e.g., real spatial distances, continuous features with meaningful scales).
    - Use **Cosine** when orientation matters more than length (e.g., text/image embeddings, TF-IDF vectors).

###### When to use cosine similarity but not Euclidean distance?
For two vectors A and B, when their cosine similarity are being defined as $cos(A,B)=\frac{A\cdot B}{\|A\|_2 \|B\|_2}$ , i.e. the cosine of angle between two vectors, we thus measure the angular distance between them, rather than the absolute magnitude, with the range being [-1,1]. When a pair of text being very different in length, but with similar content, if using Euclidean distance, one can think their distance being pretty big whereas when using cosine similarity, the angle between the two can be rather small, hence giving high similarity. In text, visual, video, image industries, when the objective has high dimensions, cosine can still retain its character of [-1,1] whereas the Euclidean distance number can be really big. 

<span style="color:#FF6961">Overall, Euclidean distance measures the absolute difference between numbers whereas the cosine distance measures the directional relative difference. </span> 

Taking an example of measuring user behavior of watching two different TV series:
	- user A's watch vector = (0,1)
	- user B's watch vector = (1,0)
It is obvious that the cosine distance between the two can be really big whereas their Euclidean distance is small. 

When measuring user A/B preference, we focus more on relative difference, hence we should be using the cosine distance whereas when we are analyzing user login frequency or activity, we should be using Euclidean distance instead as the cosine distance would think two users of vector (1,10) and (10,100) are more similar to each other.

###### Is cosine distance a strictly defined distance?
No, it is not strictly defined as it satisfies the Non-negativity & identity (strictness), symmetry but does not satisfy the triangle inequality. A use case of this question is that when reading the word vector of `comedy` and `funny` and also `happy` and `funny`, their cosine distance is < 0.3, whereas the distance between `comedy`and `happy` is 0.7. 

#### Model Evaluation Methods
#holdout #cross-validation #bootstrap
In ML algorithm design, we usually split the samples into training and test data set, where the training set is used to training the model and the test set is used to evaluate the model. In sample split and model evaluation process, we could use different sampling or evaluation methods. 

##### In model evaluation, what are the main evaluation methods, what are their pros and cons?

- **Holdout evaluation**: Holdout evaluation is the easiest way as it randomly split the original sample set into training and evaluation. For example, for a clickthrough rate prediction algorithm, we split the samples into 70 - 30%. We use the 70% data for model training and the 30% for evaluation, including ROC curve, accuracy calculation and recall rate metric evaluation. This has significant downside: the calculated final evaluation metric is highly correlated with the original data split. In order to eliminate this randomness, researchers started to use the "cross validation" idea.
- **cross-validation**: k-fold cross validation would always split the data set into k different sets that are of same counts. The method goes through all the k sample sets and always use the current subset as the evaluation set whereas the other ones are training set. usually we use k = 10.
- **Bootstrap**: 
	- Make a **fake test set** by randomly picking the same number of rows from your real test set **with replacement** (so rows can repeat and some are left out).
		- Suppose the test set has **n rows**.
		- Pick **n indices at random WITH replacement** from `0..n-1`. (Duplicates allowed; some rows wonâ€™t be picked.)
		- Those picked rows form one **fake test set**. 
	- On that fake set, compute your metric (accuracy, F1, AUC, RMSEâ€”whatever you care about).
	- Repeat steps 1â€“2 a lot (like **1,000 times**).
	- Now you have 1,000 metric values.
	    - The **average** is your central estimate.
	    - The **middle 95% range** (ignore the lowest 2.5% and highest 2.5%) is your **95% confidence interval**.
As $n$ gets large, about **36.8%** of items are â€œout-of-bagâ€ (never selected) and **63.2%** appear at least once. This is the source of the â€œ.632â€ bootstrap terminology

#### Hyperparameter tuning
#hyperparameter-tuning 
For a lot of algorithm engineers, hyperparameter tuning can be really of headache, as there is no other way other than empirically tune the parameters to a reasonable range, while it is really important for the algorithm to be effective.

##### What are some of the common ways of hyperparameter tuning?
- **grid search**: Exhaustive on a small, **low-dimensional** space. Deterministic but expensive; scales poorly. In reality, it tend to be used as a bigger search space and larger step size to find the possible range of optimal results, then to shrink the search space and find more accurate optimal solution.
- **random search**: Sample hyperparams at random (often **log-uniform** for learning rates). Much better than grid when only a few dims matter but cannot guarantee for a optimal solution.
- **Bayesian optimization**: Model â€œconfig â†’ scoreâ€ to pick promising next trials. Unlike random/grid search **donâ€™t learn** from past trials, BO **uses what youâ€™ve learned so far** to place the next (expensive) trial where itâ€™s most likely to pay off.

#### Overfit and Underfit
#overfit #underfit
This section tells how one could efficiently recognize overfit and underfit scenarios and do model improvements based on what has been identified. 

##### What is overfit and what is underfit?
- **Overfit** means that a model can be overfitting on its training data whereas on the test and new data sets, it's performing worse. 
- **Underfit** means that the model is performing illy on both training and test data sets. 

##### What are some ways to mitigate the overfit and underfit?
- **Avoid overfit**: 
	- **Data**: obtaining more data is one primitive way of solving overfit problem as more data can help the model to learn more efficient features to mitigate the impact from noise. Using rotation or expansion for image or GAN for getting more new training data.
	- **Model**: one could use less complicated / complex model to avoid overfitting. For example, in NN one could reduce the number of layers or neurons in each layer; or in decision tree, one could reduce the depth of the tree or cut the tree.
	- **Regularization**: one could use L2 regularization in model parameters to constraint the model. 
	- **ensemble method**: ensemble method is to integrate multiple models together to avoid a single model overfitting issue, such as bagging methods.
- **Avoid underfit**:
	- add more features: when there is not enough features or the features are not relevant with the sample labels, there would be a underfit. We could dig into contextual features / ID features / combination of features to obtain better results. In deep learning, factor decomposition / gradient-boosted decision tree / deep-crossing can all be used for get more features.
	- increase the complexity of model. 
	- decrease regularization parameters. 

##### L2 / L1 Regularization
#l2-regularization #l1-regularization #l2 #l1
###### Setup
Model (no intercept for simplicity):

$$\hat y_i = w\,x_i$$

**Data loss** (sum of squared errors):

$$\sum_i (y_i - w x_i)^2$$
**L2-regularized loss** (ridge):
$$\underbrace{\sum_i (y_i - w x_i)^2}_{\text{fit the data}} \;+\; \underbrace{\lambda\, w^2}_{\text{penalize big weights}}$$â€‹â€‹
- $\lambda>0$ controls the strength of the penalty (larger $\lambda$ â‡’ stronger shrinkage).
- In practice, we usually **donâ€™t penalize the bias/intercept**.
###### How L2 Penalizes the Parameter

Take derivative w.r.t. $w$ and set to 0:

$$\frac{\partial}{\partial w}\Big[\sum_i (y_i - w x_i)^2 + \lambda w^2\Big] = -2\sum_i x_i(y_i - w x_i) + 2\lambda w = 0$$

Rearrange:
$$â€‰w\big(\sum_i x_i^2 + \lambda\big) = \sum_i x_i y_i \quad\Rightarrow\quad \boxed{\,w_{\text{ridge}} = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}\,}$$â€‹â€‹â€‹
Compare to **unregularized** OLS:
$$w_{\text{OLS}} = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2}$$â€‹â€‹
L2 adds$\lambda$ to the denominator â‡’ **shrinks $w$ toward 0**.

###### Tiny Numeric Example

Data: $x=[0,1,2,3]$, $y=[0,1,2,60]$ (last point is an outlier)
- $\sum x_i^2 = 14, \sum x_i y_i = 185$
Weights:
- **OLS (no L2):** $185/14 \approx 13.214$
- **L2, $\lambda=10$:** $185/(14+10) = 185/24 \approx 7.708185$
- **L2, $\lambda=100$:** $185/(14+100) = 185/114 \approx 1.623$
As $\lambda$ grows, $w$ is **pulled toward 0**, limiting the impact of the outlier.

###### Gradient-Descent View (Weight Decay)
With learning rate $\eta$:
$$w_{\text{new}} = w_{\text{old}} - \eta\Big(\underbrace{-2\sum_i x_i(y_i - w_{\text{old}} x_i)}_{\text{data gradient}} \;+\; \underbrace{2\lambda w_{\text{old}}}_{\text{L2 shrink}}\Big)$$

The $+2\lambda w$ term is the **shrinkage** that steadily decays weights.
###### Multi-Feature Form (for reference)

For features $X\in \mathbb{R}^{n\times d}$, target $\mathbf{y}$:

$$\mathbf{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$$
###### Copy-Paste Python

```
import numpy as np

x = np.array([0,1,2,3], dtype=float)
y = np.array([0,1,2,60], dtype=float)

Sxx = np.sum(x**2)
Sxy = np.sum(x*y)

def ridge_weight(lmbda):
    return Sxy / (Sxx + lmbda)

print("w_OLS        =", Sxy / Sxx)
for lmbda in [10, 100]:
    print(f"w_ridge(Î»={lmbda}) =", ridge_weight(lmbda))
```

**Notes**
- Standardize features before using L2/L1 (esp. linear/logistic).
- Tune $\lambda$ via cross-validation.
- Do **not** penalize the bias term.

### Classical Algorithms
#### Support Vector Machine (SVM)

