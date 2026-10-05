2. Task title and overview
Potential title: Perceived heat-refuge suitability in urban streetscapes: learning from
human judgement and disagreement
2.1 Problem and motivation:
Extreme heat is becoming a growing challenge in cities, and people often make quick
visual judgments about whether a street, plaza, or public space looks suitable for
escaping high temperatures. These judgments are influenced by visible features such
as shade, vegetation, building materials, openness, and the overall configuration of the
space.
Most existing research focuses on estimating thermal comfort through physical
measurements, environmental variables, or simulation models. These methods are
useful for measuring actual thermal conditions, but they do not directly capture how
people perceive a place when they simply look at it.
This project focuses on that perceptual dimension. The goal is to build a new image
dataset that represents how suitable urban places appear as refuges from extreme heat,
while also preserving the differences in opinion between annotators.
2.2 Research question
Can a lightweight vision model predict both collective human perception and the level
of human disagreement about how suitable an urban place appears as a refuge from
extreme heat, using only a street-level image?
2.3 Novelty
Existing work on urban heat and street-view imagery often focuses on physical or
computational measurements of thermal comfort. For example, some studies use
computer vision to detect elements such as trees and urban geometry, then combine
this information with environmental variables to estimate thermal conditions.
The closest perceptual work we identified is Yang et al. (2025), who introduced the
idea of thermal affordance and asked participants to compare street-view images
based on perceived thermal comfort. Their pairwise judgments were later converted
into normalized scores.
Our project takes a different approach. Instead of reducing all human judgments to a
single label or score, we preserve the full distribution of annotations for every image.
This allows us to study not only what people generally think about a scene, but also
how much they agree or disagree.
This idea is related to previous research on learning from disagreement, where
variation between annotators is treated as meaningful information rather than
annotation noise. Our contribution is to apply this idea to perceived heat-refuge
suitability in urban environments.
3. Task definition
3.1 Inputs
The input is a single RGB street-level image showing an outdoor urban environment.
3.2 Annotations
Each image is independently evaluated using one question:
"Based only on what is visually observable, how suitable does this place appear as a
refuge from extreme heat?"
Annotators select one of five ordered categories:
1. Poor Heat Refuge
2. Limited Heat Refuge
3. Moderate Heat Refuge
4. Good Heat Refuge
5. Excellent Heat Refuge
Annotators do not see the responses of other participants.
These labels are created specifically for this project, so the original image collection
does not contain the target labels before annotation.
3.3 Outputs
The main model output will be a probability distribution over the five heat-refuge
categories rather than only a single class.
From this distribution, we can obtain:
- the most likely category
- the expected heat-refuge rating
- a measure of predicted disagreement or uncertainty
This allows the model to represent cases where human judgment is highly consistent
as well as cases where opinions are divided.
3.4 ML idea
The task can be treated as a five-class ordinal image classification problem.
We will compare two main approaches. The first will use a traditional hard label
based on the majority vote. The second will use the complete distribution of human
ratings as a soft label.
This comparison will help us understand whether preserving disagreement provides
useful information for model training and prediction.
4. Dataset and annotation plans
4.1 Data sources
Possible image sources include:
- KartaView
- Panoramax
- Wikimedia Commons
These platforms provide access to street-level or urban imagery that can be used to
build the dataset.
4.2 Image collection strategy
We plan to collect candidate images from multiple cities, regions, and urban
environments to create a dataset with enough visual diversity.
The first stage will focus on collecting a relatively large pool of images using neutral
streetscape criteria. The collection process should not intentionally search for places
that already appear "good" or "bad" as heat refuges, since this could introduce
label-related bias before annotation.
After collection, we will use pretrained computer vision models to extract simple
visual information from the images. These models may identify or tag elements such
as:
- trees
- vegetation
- streets
- sidewalks
- buildings
- open sky
- shadows
- vehicles
- urban furniture
- sunny or cloudy conditions
These tags will not be used as ground-truth heat-refuge labels. Their main purpose
will be to understand the composition of the dataset and help us identify
overrepresented visual patterns.
For example, if a very large percentage of the images contain the same combination
of characteristics, such as tree + street + sunny, we can reduce that group or collect
additional images representing less common combinations.
The objective is not to perfectly balance every possible visual feature. Instead, we
want to avoid building a dataset dominated by only a few types of urban scenes.
We will keep this preprocessing stage relatively simple. The main research
contribution should remain the human annotations and the prediction of perception
and disagreement, rather than building an overly complex automatic tagging pipeline.
4.3 Annotation procedure
Once the image collection has been reviewed and the dataset is sufficiently diverse,
human annotation will begin.
Each image will receive approximately 3 to 5 independent ratings using the same
question and five-category scale.
The annotation interface can be implemented using Label Studio or a similar
lightweight annotation platform.
Annotators will receive short written instructions explaining that they should judge
only what is visually observable in the image and should not assume information
about temperature, weather, location, or conditions that are not visible.
4.4 Human disagreement benchmark
In addition to the main dataset, we plan to create a smaller subset with significantly
more annotations per image.
Approximately 1 to 300 images will receive 15 to 20 independent ratings each.
This subset will provide a more stable estimate of the distribution of human opinions
and will be used as a Human Disagreement Benchmark.
The idea is similar to datasets such as ChaosNLI, where selected examples receive
many annotations so disagreement itself can be studied more carefully.
This benchmark will allow us to evaluate whether a model can reproduce not only the
majority human judgment, but also the actual pattern of agreement and
4.5 Licensing and data usage
The dataset will only use imagery that can legally be reused for the purposes of the
project.
For each image, we will store metadata describing its original platform, license, and
attribution requirements.
Potential sources include KartaView, Panoramax, and Wikimedia Commons, which
provide openly licensed material under licenses such as CC BY, CC BY-SA, CC0,
public-domain terms, or other compatible open licenses depending on the source.
Because licensing may vary by platform, instance, or individual image, license
information will be tracked at image level instead of assuming that every image
follows exactly the same terms.
More restrictive image sources, such as Google Street View, may be useful for
exploration but would not be included in the redistributed dataset unless their terms
clearly allow that use.
4.6 Expected dataset size and example data point
The target dataset size is approximately:
- Main dataset: 800 to 1200 images with around 5 ratings per image
- Human Disagreement Benchmark: 100 to 300 images with approximately 15
to 20 ratings per image
A simplified example of one dataset entry could look like this:
image_id: img_00842
source_platform: KartaView
license: CC-BY-SA-4.0
region: coarse city or neighborhood
capture_date: 2023-07
scene_tags: [tree, sidewalk, partial_shadow, sunny]
ratings: [3, 4, 3, 3, 4]
label_distribution:
Poor: 0.0
Limited: 0.0
Moderate: 0.6
Good: 0.4
Excellent: 0.0
majority_label: Moderate Heat Refuge
disagreement_score: 0.67
The final metadata structure may change as the dataset is developed.
5. Proposed modeling approach
5.1 Hard-label baseline
The first model will be a standard image classifier trained using the majority-vote
category for each image.
This provides a simple baseline that represents the traditional approach of reducing
multiple annotations to one final label.
Cross-entropy loss can be used for training.
5.2 Soft-label model
The second model will use the same or a similar architecture, but instead of learning
from a single category, it will learn from the complete five-class annotation
distribution.
For example, if five annotators rate an image as:
[Moderate, Moderate, Moderate, Good, Good]
the training target becomes:
[0.0, 0.0, 0.6, 0.4, 0.0]
Possible training losses include soft cross-entropy, KL divergence, or Jensen-Shannon
divergence.
The objective is to determine whether learning from the full distribution improves the
model's ability to represent collective human perception.
5.3 Human disagreement prediction
Human disagreement will be calculated from the distribution of ratings for each
image.
Possible measures include entropy, variance, or another normalized disagreement
score.
We will then compare the disagreement implied by the model's predicted distribution
with the disagreement observed in the human annotations.
This creates a second prediction problem:
Can the model recognize which images are likely to produce agreement and which
images are likely to produce divided opinions?
5.4 Candidate architectures
Initial candidate models include:
- MobileNetV3
- EfficientNet-B0
- ResNet18
These architectures are relatively lightweight, widely used, and available with
ImageNet-pretrained weights.
Using pretrained models keeps the project computationally realistic while still
providing strong image representations for fine-tuning.
We may test more than one architecture, but the focus of the project will be on
comparing hard-label and soft-label learning rather than performing an extensive
architecture search.
6. Evaluation metrics
Different metrics are needed because the project evaluates both classification
performance and the quality of predicted human distributions.
For hard-label prediction, we will use accuracy, Macro-F1 and MAE. Mean Absolute
Error is particularly useful because the five classes are ordered. Predicting Good
instead of Excellent should be treated as a smaller error than predicting Poor instead
of Excellent.
To analyze agreement between human annotators, we will consider Krippendorff's
Alpha and/or Fleiss' Kappa.
For soft-label prediction, we will use:
- Jensen-Shannon Divergence
- Cross-Entropy
- potentially KL Divergence
These metrics compare the predicted probability distribution with the human
annotation distribution.
For disagreement prediction, we will consider:
- Mean Absolute Error
- Pearson or Spearman correlation
This combination of metrics allows us to evaluate three different questions:
1. Can the model predict the most common human judgment?
2. Can the model reproduce the complete distribution of judgments?
3. Can the model identify scenes where humans are likely to disagree?
7. Non-Triviality for frontier LLM
7.1 Experimental setup
This experiment has not been run yet.
We plan to select a small set of images covering different visual conditions and
heat-refuge ratings.
A frontier multimodal LLM will receive each image together with the exact same
annotation question and five possible categories used by human annotators.
The initial experiment will use zero-shot prompting, without examples or fine-tuning.
Where possible, each image may be evaluated multiple times or using more than one
frontier model to examine consistency.
7.2 Results
TBD
7.3 Human LLM disagreement
We will compare LLM predictions against both the majority human rating and the
complete human annotation distribution.
Of particular interest are images where humans strongly disagree.
For example, an image may receive similar numbers of Moderate and Good ratings. If
the LLM produces a highly confident prediction for only one category, this may
suggest that the model does not represent human disagreement well.
The opposite case is also relevant. The LLM may show uncertainty for images where
human annotators strongly agree.
These cases can help us better understand the difference between model uncertainty
and actual human disagreement.
7.4 Distribution challenge
Our expectation is that predicting a reasonable single label may not be the most
difficult part of the task for a frontier multimodal model.
The more challenging question is whether a model can reproduce the specific
distribution of human responses for each image.
For example, two images may both have Moderate as their majority label, but one
may receive almost complete agreement while the other may be split between
Limited, Moderate, and Good.
A model that predicts only the majority category would treat these images as
essentially identical. A distribution-based model should be able to represent this
difference.
This is one of the main reasons for creating the dataset with multiple independent
annotations.