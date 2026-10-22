# Beyond the Mean Prototype: Multi-Modal Class Representations for Few-Shot Learning

Prototypical Networks describe each class with **one point**: the mean of its support embeddings. That works when a class looks the same in every example. But many real classes come in **several styles** (a letter written upright or rotated, a bird seen from the front or the side), and the mean of two styles can sit in empty space between them.

This project replaces the single mean with **several prototypes per class**, found with differentiable soft k-means, and tests when that helps.

## Key results

5-way and 20-way accuracy (%) with 95% confidence intervals over **600 test episodes** per setting, on Omniglot evaluation alphabets (never seen in training).
*Standard* = normal tasks. *Multi-style* = every image is randomly rotated with probability 0.5, so each class really has two styles.

| Condition | Setting | Pretrain + nearest centroid | ProtoNet | **Clustered prototypes** |
|---|---|:-:|:-:|:-:|
| Standard | 5-way 1-shot | 94.15 ± 0.45 | 95.41 ± 0.40 | 95.32 ± 0.40 |
| Standard | 5-way 5-shot | 98.37 ± 0.16 | 98.59 ± 0.14 | **98.74 ± 0.14** |
| Standard | 20-way 1-shot | 83.93 ± 0.42 | 86.18 ± 0.41 | **86.74 ± 0.39** |
| Standard | 20-way 5-shot | 95.05 ± 0.21 | 95.95 ± 0.19 | **96.25 ± 0.18** |
| Multi-style | 5-way 1-shot | 86.49 ± 0.66 | 90.46 ± 0.59 | 90.60 ± 0.59 |
| Multi-style | 5-way 5-shot | 96.30 ± 0.28 | 97.38 ± 0.23 | **97.59 ± 0.22** |
| Multi-style | 20-way 1-shot | 70.48 ± 0.49 | 76.10 ± 0.50 | 75.86 ± 0.49 |
| Multi-style | 20-way 5-shot | 88.71 ± 0.31 | 91.77 ± 0.26 | **92.24 ± 0.27** |

### Paired comparison: clustered prototypes vs ProtoNet

Both models are tested on exactly the same episodes, so the difference is measured episode by episode.

| Condition | Setting | Gain (points) | Real difference? |
|---|---|:-:|:-:|
| Standard | 5-way 1-shot | −0.10 ± 0.27 | no |
| Standard | 5-way 5-shot | **+0.16 ± 0.11** | yes |
| Standard | 20-way 1-shot | **+0.56 ± 0.29** | yes |
| Standard | 20-way 5-shot | **+0.30 ± 0.16** | yes |
| Multi-style | 5-way 1-shot | +0.14 ± 0.37 | no |
| Multi-style | 5-way 5-shot | **+0.21 ± 0.15** | yes |
| Multi-style | 20-way 1-shot | −0.24 ± 0.37 | no |
| Multi-style | 20-way 5-shot | **+0.47 ± 0.21** | yes |

## Findings

1. **Clustered prototypes beat ProtoNet in all four 5-shot settings**, and every one of these gains is outside the 95% confidence interval.
2. **The largest gain is on multi-style 20-way 5-shot (+0.47 points)**, exactly the case the method is designed for: classes with more than one look and enough examples to see both.
3. **At 1-shot the two methods are tied in three of four settings.** This is expected: with one example per class there is only one point, so there is nothing to cluster.
4. The gains are small in absolute terms because Omniglot accuracy is already close to its ceiling. Datasets with stronger intra-class variation are where this idea should matter more (see the follow-up project below).

## Method

**Backbone.** The standard Conv-4 network used in the few-shot literature (four blocks of 3×3 conv, GroupNorm, ReLU, max-pool), with GroupNorm instead of BatchNorm so that no method can use statistics of the query batch.

**Clustered prototype head.** For each class:
1. pick M = 2 starting points by **farthest-point selection** (deterministic and independent of the order of the support images);
2. refine them with 3 steps of **differentiable soft k-means**, with a temperature scaled to the spread of the support set;
3. score a query by the log of a mixture over the centroids, weighted by how many supports each centroid holds.

Squared distances are computed directly rather than with `cdist`, because the gradient of a Euclidean distance at zero is undefined, and a one-shot centroid sits exactly on its support point.

**Training.** Metric learners train on 20-way episodes with 1 to 5 shots and 5 queries per class, 4,000 steps, learning rate 1e-3. Half of the training tasks are multi-style, so both models see the same task distribution.

**Baselines.**
- *Pretrain + nearest centroid:* a classifier trained on all base classes, then used as a fixed feature extractor.
- *ProtoNet:* the same backbone and training budget, with a single mean prototype.
- *MAML* is also implemented in the notebook. In this run its 5-shot accuracy came out lower than its 1-shot accuracy, which points to a training problem rather than a real result, so it is left out of the conclusions.

## How to run

1. Open the notebook in Google Colab with a GPU runtime and run all cells. Omniglot is downloaded automatically through torchvision.
2. Set `FAST_DEV_RUN = True` for a one-minute end-to-end check.
3. Results, tables and plots are written to `outputs_part2/`.

## Main libraries

PyTorch · torchvision · NumPy · pandas · Matplotlib

## Related projects

- [Seeing with Words](https://github.com/SadafZehra/Multimodal_fewshot_prototypes): extends prototypes with CLIP text descriptions on CUB-200 and mini-ImageNet.
- [SCACM](https://github.com/SadafZehra/SCACM-Semantic-Clustered-Attention-Conditioning-for-Few-Shot-Generation-on-Unseen-Alphabets): the same clustering idea used for few-shot *generation*.

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
