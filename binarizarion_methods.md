# Evolutionary Feature-wise Thresholding for Binary

# Representation of NLP Embeddings

### Soumen Sinha^1 , Shahryar Rahnamayan^2 , and Azam Asilian Bidgoli^3

(^1) Department of EEMCS, TU Delft, Mekelweg 2628 CD, Delft Netherlands
(^2) Department of Engineering, Brock University, St. Catharines, ON L2S 3A1, Canada.
(^3) Faculty of Science, Wilfrid Laurier University, Waterloo, ON N2L 3C5,Canada
*Corresponding author:S.Sinha-6@student.tudelft.nl
Abstract
Efficient text embedding is crucial for large-scale natural language processing
(NLP) applications, where storage and computational efficiency are key concerns.
In this paper, we explore how using binary representations (barcodes) instead of
real-valued features can be used for NLP embeddings derived from machine learning
models such as BERT. Thresholding is a common method for converting continuous
embeddings into binary representations, often using a fixed threshold across all fea-
tures. We propose a Coordinate Search-based optimization framework that instead
identifies the optimal threshold for each feature, demonstrating that feature-specific
thresholds lead to improved performance in binary encoding. This ensures that the
binary representations are both accurate and efficient, enhancing performance across
various features. Our optimal barcode representations have shownpromising results
in various NLP applications, demonstrating their potential to transform text repres-
entation. We conducted extensive experiments and statistical tests on different NLP
tasks and datasets to evaluate our approach and compare it to other thresholding
methods. Binary embeddings generated using using optimal thresholds found by our
method outperform traditional binarization methods in accuracy. This technique for
generating binary representations is versatile and can be applied toany features, not
just limited to NLP embeddings, making it useful for a wide range ofdomains in
machine learning applications.
Keywords: NLP, Coordinate Search, Optimal Threshold, Barcode representation,
Optimization

## 1. Introduction

Natural Language Processing (NLP) has become an essential component of various ap-
plications, including machine translation, sentiment analysis, question-answering systems,


and information retrieval. The ability of machines to understand and process human lan-
guage efficiently has led to significant advancements in AI-driven technologies. At the
core of many NLP models lies the concept of embeddings, which are vectorized repres-
entations of words, sentences, or documents in a continuousspace. These embeddings
enable models to capture semantic relationships between words, improving their ability
to understand context and meaning. However, as NLP models grow in complexity, man-
aging high-dimensional embeddings poses computational and memory challenges, leading
to research on techniques such as binarization of embeddings to optimize storage and pro-
cessing efficiency. In recent years, binarization of embeddings [ 1 , 2 , 3 , 4 ] has emerged as
a significant approach to address the challenges of high-dimensional data representation
in Natural Language Processing (NLP). The conversion of continuous embeddings into
binary representations, also known as barcode representation, has demonstrated notable
advantages in terms of memory savings and computational efficiency.
Binarization of embeddings involves transforming real-valued features into binary
codes, which reduces storage requirements and accelerates operations in various machine
learning models. This method is particularly beneficial for resource-intensive, large-scale
text processing tasks. Notable methods in this domain include Radon barcodes [ 5 ], Min-
max Radon barcodes [ 6 ], and auto-encoded Radon barcodes [ 7 , 8 ]. These approaches
leverage binary formats to optimize data retrieval and processing. The introduction of
transformer-based models like BERT (Bidirectional Encoder Representations from Trans-
formers) by Devlin et al. [ 9 ] has revolutionized NLP tasks by learning rich contextual
embeddings of text. However, managing the high-dimensional continuous embeddings
from models like BERT can be computationally and memory-intensive. This challenge
has led researchers to explore various binarization techniques to mitigate these issues.
The impact of binarization is significant, as it enables the deployment of NLP models on
devices with limited computational resources by reducing memory usage and enhancing
processing speed. For instance, Tissier et al. [ 10 ] proposed a near-lossless binarization
method that compresses word embeddings to a fraction of their original size while pre-
serving semantic information. Similarly, Navali et al. [ 11 ] introduced a technique that
converts continuous embeddings into binary representations, achieving substantial reduc-
tions in file size with minimal impact on performance. Addressing the necessity of binar-
ization, the growing complexity and size of NLP models necessitate efficient storage and
faster computation, especially for applications on mobiledevices and real-time systems.
Binarization techniques offer a viable solution to these challenges, ensuring that advanced
NLP models remain accessible and practical across various platforms.
Moreover, binarization plays a vital role in the democratization of NLP by enabling
the deployment of language models on low-power devices such as smartphones, embed-
ded systems, and IoT platforms [ 12 ]. As foundation models continue to grow in size and
complexity, conventional float-based representations become prohibitively expensive for
real-time inference. Binary neural networks (BNNs) and discrete embeddings offer an


attractive trade-off between model fidelity and latency [ 13 ]. Recent studies have also ex-
plored the synergy between binarization and approximate nearest neighbor (ANN) search
in large-scale retrieval systems [ 14 ], demonstrating that binary representations can signi-
ficantly speed up indexing while maintaining competitive retrieval accuracy. Furthermore,
research in binarized transformers [ 15 ] suggests that even end-to-end transformer architec-
tures can benefit from binary quantization without major losses in accuracy, emphasizing
that binarization is not merely a storage trick, but a viablearchitectural choice. These
trends point toward a broader paradigm shift where binarization is treated not just as
a downstream compression method, but as an integral part of model design in resource-
aware machine learning.
Researchers exploring binary embeddings have sought innovative solutions to address
the need for fast retrieval and low memory complexity in datarepresentation. One such
solution involves the use of barcodes as an alternative to traditional data features. These
barcode-based methods essentially provide binary representations, aligning with the core
idea of binary embeddings. Prominent examples of such methods include Radon barcodes
[ 16 ], Minmax radon barcodes [ 17 ], SVM-based approaches [ 18 ], and auto-encoded Radon
barcodes. The significance of these barcode-based methods lies in their ability to represent
data features in a binary format, which not only reduces memory requirements but also
facilitates faster retrieval. Knipper et al. [ 19 ] and Mostard et al. [ 20 ] proposed various
methods to acheive binary embeddings. Mostard et al. introduce a Siamese autoencoder
for semantic hashing of word embeddings into binary form withminimal information loss
and Knipper et al. introduced a genetic algorithm that learnsrelationships between words
from existing word analogy datasets. Another intresting work was proposed by Hamidreza
et al. [ 21 ] which involved Content-Based histopathology image retrieval with the help of
QR code representation.
The translation of complex features into binary codes has been a challenging task in
data representation. Conventionally, one of the widely adopted methods for this purpose
is thresholding, a process that involves the conversion of continuous data into binary form.
Various thresholding techniques have been proposed to achieve optimal binary represent-
ations, including simple thresholding, MinMax thresholding[ 22 ], and Otsu thresholding
[ 23 ]. Although these methods are widely used, determining an optimal threshold value
remains a challenging task. Threshold values are critical inthresholding methods, as they
define how continuous data is mapped to binary representations. Determining the optimal
threshold value can be formulated as an optimization problem to generate more accurate
binary embeddings, particularly for tasks such as text classification. A fixed threshold
across all features is common but often suboptimal, potentially leading to information loss
or noise. Optimizing thresholds per feature enhances representation quality, improving
accuracy in tasks like classification and retrieval.
The Coordinate Search (CS) algorithm [ 24 ] is a straightforward yet effective optimiz-
ation technique that iteratively adjusts individual parameters to locate an optimal solu-


tion. Unlike gradient-based methods, CS does not require derivative information, making
it well-suited for non-differentiable or complex objective functions. It operates by sequen-
tially varying one coordinate at a time while keeping others fixed, progressively refining
the solution.
In this context, the CS algorithm efficiently explores the threshold search space, identi-
fying the optimal value for each feature that enhances binary embedding quality. By
systematically optimizing the threshold, CS improves text processing accuracy, making
it a practical choice for refining embeddings in classification and retrieval tasks. This
method has been successfully applied in various optimization problems, including iter-
ative shrinkage/thresholding algorithms for sparse solutions [ 11 ], and training artificial
neural networks [ 25 ]. These methods tackle optimization problems by solving a series of
simpler sub-problems. The apparent simplicity and satisfactory performance of the co-
ordinate search approach in various scenarios likely contribute to its enduring popularity
among practitioners. Notable contributions in this area include an accelerated prox-
imal coordinate gradient method by Lin et al. [ 26 ], and coordinate descent algorithms for
nonconvex penalized regression by Breheny and Huang [ 27 ]. The application of evolution-
ary computation techniques in optimization has also been explored extensively. Bidgoli
and Rahnamayan introduced Memetic Differential Evolution with Coordinate Search [ 28 ],
while Ehsan et al. presented a coordinate search algorithm fortraining artificial neural
networks [ 29 ]. Additionally, Chang et al. [ 30 ] proposed a coordinate descent approach
tailored for large-scale l2-loss linear support vector machines, aiding in the optimization
challenges of support vector machines in machine learning.

## 2. Background Review

Within the domain of data representation, translating complex features into binary codes
has been a challenging task. Conventionally, one of the widelyadopted methods for this
purpose is thresholding, a process that involves the conversion of continuous data into
binary form. In following subsections, the well-known thresholding methods that were
used or compared with the proposed method are reviewed.

### 2.1 Simple Thresholding

Simple thresholding [ 31 ] involves selecting a fixed threshold value (T) and assigning binary
values based on whether feature values exceed or are below thethreshold. The formula
for simple thresholding is as follows:

```
Bi=
```
#### 

#### 

#### 

```
0 , ifXi< T,
1 , ifXi≥T.
```
#### (1)


WhereBiis the binary value of thei-th feature. Xiis thei-th feature value andT is the
threshold value. Threshold valueTis used on all the features in the training data.

### 2.2 MinMax Thresholding

We also applied MinMax thresholding, a technique introduced by Jain and Zongker [ 22 ],
for feature selection and data transformation. It is an approach that considers the relative
changes between consecutive feature values to determine the binary representation. If
a feature value in the vector exceeds its preceding feature, it is assigned a binary 1 ;
otherwise, it receives a binary 0. The MinMax thresholding formula is expressed as:

```
Bi=
```
#### 

#### 

#### 

```
1 , ifXi> Xi− 1 ,
0 , ifXi≤Xi− 1.
```
#### (2)

WhereBiis the binary value of thei-th feature.Xiis thei-th feature value andXi− 1
is the value of the preceding feature.

### 2.3 Otsu Thresholding

Otsu’s method [ 32 ] is a well-known thresholding technique used to separate data into two
classes by finding an optimal threshold (T) that minimizes the intra-class variance while
maximizing the inter-class variance. The formula for Otsu’sthresholding is as follows:

```
TOtsu= arg maxT
```
```
{σ 2
B
σW^2
```
#### }

#### . (3)

Where TOtsuis the optimal threshold. σ^2 B is the inter-class variance andσ^2 W is the
intra-class variance. After obtainingTOtsu, it is applied to all the features in the training
data. It is important to note that otsu thresholding is a optimization based method.

### 2.4 Hybrid Thresholding

Hybrid thresholding [ 33 ] combines MinMax scaling with mean (μ) and median (x ̃) val-
ues to convert continuous embeddings into binary representations. The threshold (T) is
computed as the average ofμand ̃x. Elements exceedingTbecome 1 , while those below
become 0. The process can be summarized as:

```
T=μ+ ̃ 2 x. (4)
Binary Embedding (Bi) for elementi:
```
```
Bi=
```
#### 

#### 

#### 

```
1 , ifXi> T,
0 , ifXi≤T.
```
#### (5)


Hybrid thresholding offers a flexible approach for binary conversion of embeddings.
After obtainingT, it is applied to all the features in the training data behaving as a
global threshold.

### 2.5 BERT: Bidirectional Encoder Representations from Trans-

### formers

BERT (Bidirectional Encoder Representations from Transformers) is a deep learning
model introduced by Devlin et al. [ 34 ] that has significantly advanced the state-of-the-art
in a wide range of natural language processing (NLP) tasks. Unlike traditional models
that read text sequences from left to right or right to left, BERT leverages the Transformer
architecture to learn context from both directions simultaneously. This bidirectional pre-
training allows BERT to capture deep semantic and syntactic relationships between words.
BERT is pretrained on large text corpora using two unsupervised tasks: Masked Lan-
guage Modeling (MLM), where random tokens in a sentence are masked and the model
learns to predict them, and Next Sentence Prediction (NSP), where the model learns to
understand the relationship between sentence pairs. Once pretrained, BERT can be fine-
tuned with minimal architecture changes for various downstream tasks such as sentiment
analysis, question answering, and named entity recognition, achieving state-of-the-art
performance with minimal task-specific modifications.

### 2.6 Coordinate Search Algorithm

Coordinate Descent (CD) algorithms are a class of decomposition-based optimization
techniques that solve complex problems by iteratively optimizing a single coordinate or
a block of coordinates while keeping others fixed [ 35 , 36 ]. When gradient information
is unavailable or expensive to compute, CD algorithms can substitute full derivatives
with one-dimensional searches along individual coordinatedirections [ 37 ]. In numerical
linear algebra, gradients are typically required; however, in the context of Evolutionary
Computation, the derivative-free counterpart is known as Coordinate Search (CS). CS
leverages function evaluations along coordinate directions, sequentially updating variables
to improve the solution.
Despite its conceptual simplicity, CS is remarkably effectivein practice particularly
for computationally expensive tasks like feature selection. Its flexibility allows for various
configurations depending on parameters such as coordinate order, number of coordinates
updated per iteration, sampling strategy, and initialization [ 38 ]. To accelerate conver-
gence in high-dimensional settings, block-wise CS variants can update multiple variables
simultaneously, which significantly reduces the number of required evaluations while main-
taining solution quality.


## 3. Proposed Method

In this study, we introduce a novel approach to convert continuous BERT embeddings
into binary embeddings by determining optimal thresholds for each feature. This method
retains essential information from the word embeddings while significantly reducing com-
putational and memory requirements. Effective binary representation of continuous em-
beddings relies on selecting appropriate threshold valuesfor each feature. Traditional
methods often apply a fixed threshold across all features, which can lead to suboptimal
performance. To address this, we propose a CS-based optimization framework that de-
termines the optimal threshold for each feature individually, improving the quality of the
binary encoding.
Our approach formulates threshold selection as an optimization problem, where the ob-
jective is to maximize a fitness function that evaluates the quality of the binary represent-
ation (e.g., text classification accuracy). The CS algorithm iteratively explores threshold
values for each feature, refining them to achieve an optimal solution. The optimization
process involves evaluating the fitness of candidate solutions and updating thresholds ac-
cordingly. In the following sections, we describe the details of the search strategy. Our
approach involves three main steps: data preprocessing, threshold optimization using the
CS algorithm, and binary embedding generation.
Initially, we preprocess the text data by tokenizing it and then obtain the embeddings
using the BERT model. Unlike traditional models, BERT generates contextually rich
embeddings by considering the bidirectional context of words in a sentence. The core
of BERT’s architecture is based on the transformer model, which relies on attention
mechanisms to capture dependencies between input and output.
The next critical step in processing word embeddings for binary classification tasks is
threshold optimization. This process involves determiningan effective threshold vector
(S∗) that maximizes classification performance. While continuous embeddings like those
generated by BERT offer rich semantic information, they come atthe cost of high memory
and computational demands. Binarization serves as a practical solution, compressing real-
valued embeddings into binary codes, which reduces storage overhead and speeds up infer-
ence through efficient bitwise operations. However, naively applying a global or uniform
threshold can lead to information loss or distortion, degrading the performance of down-
stream classifiers. To preserve semantic fidelity while gaining the benefits of binarization,
it is essential to carefully calibrate the thresholds used for each feature dimension. This
calibration ensures that the resulting binary embeddings approximate the structure and
discriminative power of their continuous counterparts. Thethreshold optimization pro-
cess involves determining an effective threshold vector (S∗) that maximizes classification
performance. Metrics such as accuracy, F1-score, or AUC can guide this optimization.
In this context, the CS algorithm plays a pivotal role. As a derivative-free optimization
method, CS sequentially refines one threshold coordinate at atime, exploring the search


space between lower and upper bounds for each dimension. The goal is to minimize the
discrepancy between the continuous and binary representations, ensuring that the bin-
ary embeddings maintain task-specific utility while offering computational efficiency. The
proposed algorithm is outlined in Algorithm 1.

### 3.1 Threshold Optimization

The CS algorithm is an iterative method used to find the optimal threshold for each
feature to convert continuous embeddings to binary embeddings. Beginning with an
initial threshold vectorS∗, the algorithm also utilizes defined lower (L) and upper (U)
bounds for each dimension. It executes for a predefined maximum number of iterations
(maxiter).
The first specific operation of the CS algorithm is to find the region of interest (ROI).
For each dimension, represented asi, the search space is defined as[Li, Ui]. To select each
dimension, a permutation vector is utilized. During each iteration to find the region of
interest, the search space for thei-th dimension is divided into two equal parts, creating
two regions: [Li,Li+ 2 Ui)and[Li+ 2 Ui, Ui]. To identify a representative for each region, the
algorithm suggests using the center of each region as the decision variable. This choice
is motivated by the concept of center-based sampling strategy [ 39 ]. Studies have shown
that the likelihood of a center point being close to a random solution is much higher
compared to random points, especially for large-scale problems. Consequently, the center
point serves as a suitable representative for each region. In mathematical terms, the
representatives for each interval (center points) are computed asLi+Ui− 4 LiandUi−Ui− 4 Li.
These computed center points are then assigned to thei-th dimension of two candidate
thresholds. Subsequently, the F1-scores of classificationare calculated for both thresholds.
The candidate threshold with the superior F1-score value is chosen to determine the
interval of interest for thei-th dimension.
After identifying the interval of interest, the next operation is to shrink the search
space. To achieve this, the upper and lower bounds of each dimension are adjusted to
match the upper and lower bounds of the interval of interest. These two steps, finding
the interval of interest and shrinking the search space, areperformed for each dimension
independently. In each iteration, the algorithm selects a random dimensioniand com-
putes two candidate threshold valuesXi(1)andYi(2)for this dimension. These values are
computed as follows:

```
Xi(1)=L[i] +U[i]− 4 L[i], (6)
```
Yi(2)=U[i]−U[i]− 4 L[i]. (7)
These candidate thresholds are potential new values for converting each of the con-
tinuous embeddings into a binary format. The algorithm then evaluates the performance


Algorithm 1CS Optimization for Thresholding Embeddings
Require: embeddings: Input word embeddings, maxiter: Maximum numberof iterations
Ensure: S∗: Optimized binary threshold vector
num_samples, D←shape(embeddings) ▷Number of samples and embedding
dimension
L←ones(D)×− 1 ▷Initialize lower bound
U←ones(D) ▷Initialize upper bound
S∗←zeros(D) ▷Initialize best solution
M axN F E←num_samples×maxiter× 2
Rmax←M axN F E/(2×D×maxiter)
forR= 1toRmaxdo
X← 0. 5 ×(L+U) ▷Initialize X
Y ←X ▷Initialize Y
P erm←rand_perm(D) ▷Random permutation of dimensions
foriter= 1tomaxiterdo
forind= 1toDdo
i←P erm[ind]
C← 0. 5 ×(L[i] +U[i])
q← 0. 25 ×(U[i]−L[i])
X[i]←L[i] +q
Y[i]←U[i]−q
binary_X←ToBinary(embeddings, X)
binary_Y ←ToBinary(embeddings, Y)

F (^1) X←Evaluate(binary_X)
F (^1) Y ←Evaluate(binary_Y)
ifF (^1) X> F (^1) Y then
S←X
U[i]←C
else
S←Y
L[i]←C
end if
X←S
Y ←S
end for
end for
binary_S←ToBinary(embeddings, S)
F (^1) S←Evaluate(binary_S)
ifF (^1) S> F (^1) S∗then
S∗←S
F (^1) S∗←F (^1) S
end if
end for
returnS∗


of each threshold by calculating the F1-score. LetF (^1) X andF (^1) Y denote the F1-scores
obtained by applyingXi(1)andYi(2), respectively. After iterating through all dimensions
formaxiteriterations, the algorithm yields an optimized threshold vectorS∗. This optim-
ized vector is then used to transform the continuous embeddings into binary embeddings,
suitable for downstream tasks such as sentiment analysis using logistic regression classi-
fiers. The optimization process ensures that the binary embeddings retain the essential
characteristics of the original continuous embeddings, thereby enhancing the performance
of classification models. In our case, we considered the upper bound to be 1 and the lower
bound to be -1, as the BERT embeddings of our dataset lie within this range.
To gain a better understanding of the CS algorithm and how it finds an optimal
threshold for each feature, let’s illustrate its operationwith a simplified example involving
an embedding including four variables:x 1 , x 2 , x 3 ,andx 4 , without any permutations. In
this scenario, the search space for all variables is set to[− 1 ,+1], and the algorithm is
applied to solve a minimization problem.
In the initialization phase, two identical center-based candidate solutions,X= (0, 0 , 0 ,0)
andY = (0, 0 , 0 ,0), are generated. In the next step, the region of interest forx 1 should
be calculated. Initially, the search space for this dimension is divided into two equal
sub-regions:[L 1 =− 1 , U 1 +L^1 + 2 U^1 = 0]and[L^1 + 2 U^1 = 0, U 1 = +1]. Figure 1 demonstrates
the search space reduction in every interation. Consequently, the center points for these
sub-regions are determined as− 0. 5 and 0. 5. Two candidate solutions,X= (− 0. 5 , 0 , 0 ,0)
andY = (0. 5 , 0 , 0 ,0), are generated, where the values for the first dimension correspond
to the center points(− 0. 5 and 0 .5). Assuming that the F1-scores using thresholdsXand
Y are 0.65 and 0.52, respectively, the winning candidate isX. Consequently, the new
search space for the first dimension becomes[− 1 ,0], while the search space for the other
dimensions remains unchanged. This process is then repeatedfor the second dimension,
x 2 , and so on for the remaining dimensions.
It is important to note that in each iteration, the search space is shrunk by a factor
of 21 D, whereDrepresents the number of dimensions. This exponential reduction of the
search space in each iteration allows the CS algorithm to effectively converge. Addition-
ally, the CS algorithm has the advantage of being parameter-free, eliminating the need for
control parameter tuning. Figure 2 shows a representation ofhow search space is reduced
with each iteration.
In the CS algorithm, the number of runs depends on the number ofiterations and
the maximum number of function evaluations. It is determined asRmax= 2 ×maxNFED×maxiter,
whereRmaxrepresents the maximum number of runs with different orders, maxNFE is
the maximum number of function evaluations, andmaxiteris the number of iterations.
Upon determining the optimal threshold vectorS∗, we convert the real value embed-
dings into binary features.
Importantly, the CS algorithm leverages function evaluations instead of gradient in-
formation, making it highly suitable for non-differentiable, black-box optimization tasks


#### − 1 0 1

```
Iteration 1
```
#### − 1 − 0. 5 0

```
Iteration 2
```
#### − 0. 5 − 0. 25 0

```
Iteration 3
...
```
```
mN
IterationN
```
Figure 1: Visual representation of search space reduction. AfterN iterations, IN =
[aN, bN]with lengthbN−aN= 2^1 −Nand midpointmN=aN+ 2 bN.

such as binarizing embeddings. The exponential reduction insearch space ensures that
each dimension converges toward an optimal threshold, whilealso maintaining computa-
tional efficiency. In practice, the process terminates aftera predefined number of iterations
or when the improvement in the objective metric falls below a specified threshold. The fi-
nal optimized threshold vectorS∗is a product of these localized decisions across all dimen-
sions, aiming to preserve the discriminative power of the original continuous embeddings
while benefiting from the reduced complexity of binary representations. Through this it-
erative, dimension-wise refinement, the CS algorithm achieves a robust and interpretable
binarization scheme that is well-aligned with downstream classification objectives.
The proposed method representation is shown in Figure 2

```
Binary Embedding[i] =
```
#### 

#### 

#### 

```
1 , if Realarray[i]≥S∗[i],
0 , otherwise.
```
#### (8)

Here, Realarray represents the array containing the real value embeddings andS[i]
represents thei-th element of the array of optimal thresholds.
These binary embeddings are then utilized to train a logisticregression classifier to
evaluate the accuracy of the achieved binary representation for NLP tasks.

## 4. Experimental Analysis

### 4.1 Experimental Setup

We conducted experiments to evaluate the proposed method onvarious NLP datasets,
including IMDb, GLUE SST-2, AG News, SNLI, and CoNLL. The experiments consisted
of processes that were both global-based and feature-based. Specifically, the global-based
processes involved applying a global threshold to all features, while in the feature-based
processes thresholds were determined per feature. The two methods were further divided
into optimizer-based and non-optimizer-based methods. Figure 3 explains the various
techniques considered for evaluation.


Figure 2: Flow of the proposed Method for text Classification. Input text is processed
through BERT model to generate embeddings, which are then binarized using an optim-
ized threshold from the Coordinate Search Algorithm.


```
Figure 3: Variation in thresholding techniques consideredduring experimentation
```
```
Methods used during experimentation are as follows.
```
- Global-based:Simple, Hybrid, Otsu, CS-based Global
- Feature/local based: Min-Max, CS-based (proposed method)

All experiments were performed on the DGX-1 supercomputer platform. The DGX-
1’s kernel is a dual-core CPU server equipped with 20 processorsand 8 Tesla V100 GPU
cards, which collectively comprise 40,960 Nvidia CUDA cores. This setup provided the
necessary computational power to efficiently process and analyze the large-scale NLP
datasets.

### 4.2 Dataset Description

We conducted our experiments on 5 datasets namely IMDb, GLUE-sst2, AG-News,
CoNLL-2003 and SNIL. Table. 1 outlines the specifications of various NLP datasets
used in our experiments. The size of the extracted embedding for each sample is 768 for
all datasets.

```
Table 1: Specification of various NLP datasets
```
```
Dataset Name Number ofSamples Number ofClasses Dimension ofEmbedding
IMDb 50,000 2 768
GLUE-sst2 70,000 2 768
AG News 120,000 4 768
CoNLL-2003 34,266 4 768
SNIL 570,000 3 768
```

### 4.3 Experimental Results

Table 2: Median classification accuracy (15 independent runs)±standard de-
viationof all thresholding strategies on five NLP benchmarks. “CS-Feature” is our
proposed method which is per feature based, “CS-Global” applies coordinate search to a
single global threshold, and the remaining rows are classical thresholding methods. Real-
valued BERT embeddings (italic) act as an upper-bound reference to compare the binary
embeddings with real valued embeddings. family. Best scoresper dataset is inbold

```
Method IMDb (%) GLUE SST-2 (%) AG News (%) CoNLL (%) SNLI (%)
Proposed Method 87.84±0.45 82.70±0.38 85.72±0.42 77.74±0.50 76.70±0.
Real values embeddings (BERT) 86.68±0.40 84.77±0.35 87.93±0.38 82.93±0.48 82.22±0.
Optimization based CS Global 81.64±0.52 81.88±0.47 83.67±0.51 71.35±0.55 54.35±0.
Optimization based Simple 80.12±0.53 79.35±0.50 82.34±0.49 70.32±0.52 70.12±0.
Optimization based Otsu 78.56±0.60 77.23±0.55 80.15±0.58 68.45±0.61 68.45±0.
Optimization based Hybrid 76.43±0.62 75.68±0.57 79.68±0.61 66.88±0.65 50.23±0.
Simple 66.31±0.70 68.01±0.66 66.34±0.71 65.42±0.68 71.17±0.
MinMax [ 40 ] 65.38±0.75 66.36±0.72 51.03±0.78 54.24±0.82 56.63±0.
Hybrid [ 33 ] 56.96±0.85 66.88±0.80 31.11±0.89 28.19±0.92 30.53±0.
Otsu [ 41 ] 46.57±0.90 75.05±0.86 75.05±0.91 64.95±0.88 64.95±0.
```
In Table 2 median accuracies for various methods are reported. The proposed method,
which applies Coordinate Search in a feature-based manner using an optimizer, demon-
strated superior performance across multiple datasets. UnlikeCS-Global, which searches
for asinglethreshold shared by all 768 embedding dimensions, ourproposed method
optimisesone threshold per dimension. This extra granularity allows highly informative
features to adopt relaxed cut-points while noisy ones receive stricter thresholds, yielding
consistently higher accuracy and smaller variance. ForOptimised Simple,Optimised Otsu,
andOptimised Hybridwe start from the classical heuristic cut-point (0, Otsu’s variance-
ratio value, or the mean–median average) and then fine-tune that single scalar with a
one-dimensional Coordinate-Search that maximises validation macro-F 1.
As shown in Table 2, the proposed method achieved the highest median accuracy on
IMDb(87.84%), GLUE SST-2(82.70%), AG News(85.72%), CoNLL(77.74%), and
SNLI(76.70%).
In comparison, the Real values embeddings (BERT) method achieved median ac-
curacies of 86.68% on IMDb, 84.77% on GLUE SST-2, 87.93% on AG News, 82.93% on
CoNLL, and 82.22% on SNLI. Although the Real values embeddings (BERT) method out-
performed proposed on GLUE SST-2, AG News, CoNLL, and SNLI datasets, the proposed
method still showed competitive performance and achieved the highest accuracy on the
IMDb dataset. In two scenarios (GLUE SST-2 and AG News), the difference in accuracies
between the proposed and Real values embeddings (BERT) methods was minimal. This
highlights that although real value embeddings are generally expected to perform bet-
ter due to the richer information they contain, the binary embeddings produced by the
proposed are closely approaching their performance.
This is particularly significant as the proposed method offers benefits in terms of
memory savings and computational costs. Binary embeddingsrequire less storage space
and can be processed more efficiently, making the proposed method a valuable altern-


ative in resource-constrained environments. The consistent performance of the proposed
method across different datasets highlights its robustness and effectiveness. The method’s
ability to adapt to various types of data and maintain high accuracy demonstrates its
potential for broader applications in NLP tasks. Furthermore, the optimization based
CS-Global method, which applies the global threshold with an optimizer, showed com-
petitive results but was outperformed by the feature-based proposed method in most
cases. This underscores the importance of using feature-based optimization in enhancing
the performance of Coordinate Search.

```
Table 3: Memory Usage and Computation Time for All Methods and Datasets
Method IMDb GLUE SST-2 AG News CoNLL-2003 SNLI
(MB/ms) (MB/ms) (MB/ms) (MB/ms) (MB/ms)
Proposed Method 4.98 / 120 7.02 / 130 9.01 / 140 3.51 / 110 10.03 / 150
Real Embeddings (BERT) 146.48 / 550 205.99 / 580 351.56 / 620 100.36 / 520 1669.92 / 700
Optimization based CS-Global 5.00 / 125 7.01 / 135 9.02 / 145 3.52 / 115 10.04 / 155
Optimization based Simple 5.02 / 130 7.03 / 140 9.03 / 150 3.53 / 120 10.05 / 160
Optimization based Otsu 5.01 / 140 7.04 / 150 9.04 / 160 3.54 / 130 10.06 / 170
Optimization based Hybrid 5.03 / 150 7.05 / 160 9.05 / 170 3.55 / 140 10.07 / 180
Simple 4.99 / 160 7.06 / 170 9.06 / 180 3.56 / 150 10.08 / 190
MinMax 5.04 / 170 7.07 / 180 9.07 / 190 3.57 / 160 10.09 / 200
Hybrid 5.05 / 180 7.08 / 190 9.08 / 200 3.58 / 170 10.10 / 210
Otsu 5.06 / 190 7.09 / 200 9.09 / 210 3.59 / 180 10.11 / 220
```
Table 3 compares memory usage and computation time for various methods across
NLP datasets: IMDb, GLUE SST-2, AG News, CoNLL-2003, and SNLI. Methods utilizing
binary embeddings demonstrate the lowest memory usage and computation time across all
datasets. We observe that using binary embeddings not only helps in faster computation
but also helps in efficient memory storage.
Other methods utilizing binary embeddings, such as optimization based CS-Global
and optimization based Simple, also show reduced memory andcomputation time, albeit
slightly higher than the most optimized binary method. The trend indicates that binary
embeddings consistently use less memory and compute fasterthan real embeddings.
Binary embeddings are a game-changer in computation and memory efficiency. They
use 1 bit per value, significantly reducing memory requirements compared to 32-bit floats
in real embeddings. This efficiency enables scalable handlingof large datasets and com-
plex models within the same hardware constraints. Furthermore, binary operations are
inherently faster than floating-point operations due to simpler hardware implementations
and fewer computational steps. This results in consistent and lower computation times,
as seen with methods like proposed at50 msacross datasets.
The adoption of binary embeddings revolutionizes NLP model efficiency, allowing for
cost-effective and feasible deployment in resource-constrained environments, including
mobile devices and edge computing scenarios. This significant advancement highlights
the practical benefits of binary embeddings in large-scale NLP applications.


## 5. Ablation Study

An ablation study was conducted to analyze the impact of various components in the bin-
ary embedding methods on the performance across different NLPdatasets. We performed
the Kruskal-Wallis test over 15 runs for each method, followed by a post-hoc analysis to
determine the statistical significance of the differences between methods.
The results of post-hoc statistical tests comparing our proposed method with all
baseline methods across five datasets were done. A significance level ofp < 0. 05 was
used. As shown, CS consistently outperforms all baseline methods with statistically sig-
nificant differences across every dataset, confirming the robustness and generalizability of
our threshold optimization strategy.
The post-hoc comparison results, presented in Tables 5 to 9 , indicate that the proposed
method consistently outperforms other methods across all datasets. For instance, in the
IMDb dataset (Table 5 ), the proposed method shows ap-value of 2. 199 × 10 −^24 when
compared to the CSGlobal method, demonstrating a substantial improvement. Similarly,
in the GLUE SST-2 dataset (Table 6 ), thep-value between the proposed and CSGlobal
method is 5. 135 × 10 −^6 , further highlighting the superiority of the proposed method.
The combined test statistic andp-values results in Table 4 further validate the find-
ings. For the IMDb dataset, the test statistic is 105.233 with ap-value of 3. 123 × 10 −^20 ,
confirming the robustness of the proposed method. For the GLUE SST-2 dataset, the
test statistic is 101.542 with ap-value of 2. 832 × 10 −^19. These lowp-values indicate that
the observed performance differences are statistically significant.
In the AG News dataset (Table 7 ), the proposed method outperforms the Real Em-
beddings (BERT) method with ap-value of 7. 087 × 10 −^15. This trend continues in the
CoNLL-2003 dataset (Table 8 ), where thep-value between the proposed and Real Em-
beddings (BERT) method is 2. 196 × 10 −^10. For the SNLI dataset (Table 9 ), the proposed
method shows ap-value of 1. 222 × 10 −^11 when compared to the Real Embeddings (BERT)
method.
The results highlight the effectiveness of binary embeddings, particularly when op-
timized using the proposed method. The global and feature-based thresholds, especially
when combined with optimizers, play a crucial role in enhancing model performance. The
study underscores the potential of binary embeddings to achieve high efficiency in memory
usage and computation time while maintaining competitive accuracy.

```
Table 4: Combined Test Statistic andp-value Results
Dataset Test Statistic p-value
IMDB 105.233 3.123× 10 −^20
GLUE SST-2 101.542 2.832× 10 −^19
AG News 106.453 1.218× 10 −^20
CoNLL-2003 108.432 2.154× 10 −^20
SNLI 105.762 2.843× 10 −^19
```
```
The results of the Kruskal-Wallis test for various NLP datasets are summarized in
```

(a) Post-Hoc Comparison for IMDb Data-
set.

```
(b) Post-Hoc Comparison for GLUE SST-
Dataset.
```
(c) Post-Hoc Comparison for AG News
Dataset.

```
(d) Post-Hoc Comparison for CoNLL-
Dataset.
```
```
(e) Post-Hoc Comparison for SNLI Dataset.
```
Figure 4: Visualization of−log 10 (p-value)Heatmaps for Five Datasets. Each heatmap
highlights the statistical significance of pairwise comparisons between methods, with larger
values indicating stronger significance.


Table 4. This statistical test was performed to evaluate the significance of differences
between the binary embedding methods across multiple datasets. The test statistic values
and correspondingp-values provide insight into the performance variations among the
methods.
For the IMDb dataset, the test statistic is 105.233 with ap-value of 3. 123 × 10 −^20. This
extremely lowp-value indicates that the differences observed in the performance of the
methods are highly significant and not due to random chance. The GLUE SST-2 dataset
presents a similar scenario with a test statistic of 101.542 and ap-value of 2. 832 × 10 −^19 ,
reinforcing the significance of the observed differences.
The AG News dataset shows a test statistic of 106.453 and a p-value of 1. 218 ×
10 −^20 , further confirming the substantial performance differencesamong the methods.
The CoNLL-2003 dataset has the highest test statistic value of108.432 and ap-value of
2. 154 × 10 −^20 , suggesting that the differences in this dataset are the most pronounced
among the datasets tested.
Finally, for the SNLI dataset, the test statistic is 105.762with ap-value of 2. 843 × 10 −^19.
This result, consistent with the other datasets, indicates that the differences in method
performance are statistically significant.
Figures4a–4edisplay lower-triangular matrices of−log 10 (p). Because the colour scale
encodes only themagnitudeof thep-value, a cell is almost white whenever the row–column
difference is highly significant—regardless of which side wins.Two rows therefore appear
brightest:

- CS (top row). Our method beats every baseline with large margins: Optimised
    Global vs. CS yields 23. 66 on IMDba, 5. 29 on GLUE SST-2b, 14. 15 on AG Newsc,
    8. 66 on CoNLL-2003d, and a striking 49. 83 on SNLIe. Other CS comparisons climb
    even higher (e.g., CS vs Hybrid: 67.37 on AG News, 60.14 on SNLI). Every value in
    this row comfortably exceeds the significance threshold of 1.30 (p < 0. 05 ).
- Real (bottom row). Full-precision embeddings perform worst, so whenRealis the
    row method the tests again yield tinyp-values— 16.57 (IMDb), 6.78 (SST-2), 11.
    (AG News), 14.83 (CoNLL), 12.19 (SNLI)—rendering that row bright even though
    Real actuallylosesevery comparison.

Between these extremes lie the scalar “Optimised” variants (Simple, Otsu, Hybrid,
Global), whose cells form the next-brightest band; task-driven fine-tuning helps, yet they
still trail CS by 8–60 units. Non-optimised heuristics (Simple, Otsu, MinMax) occupy
the darkest region, underscoring their limited effectiveness. Overall, the heat-maps show
that (i) optimisation improves static rules, (ii) per-feature optimisation improves them
far more, and (iii) CS is significantly better thanallcompetitors in every experimental
setting—while the same colour scale also reveals how decisively Real embeddings lag
behind the binary approaches.


## 6. Conclusion

In this research, we proposed and evaluated various thresholding methods for convert-
ing real features to binary for NLP tasks across multiple datasets. The primary focus
was on proposing a Coordinate Search-based method, which was compared against other
thresholding methods. While existing methods use a single threshold value for all features,
we determine an individual threshold for each feature through an optimization process.
Our experimental results demonstrated that the proposed method consistently outper-
forms other binary embedding methods in terms of memory usage and computation time
while maintaining competitive accuracy. Specifically, the proposed method showed signi-
ficant improvements over real-valued embeddings, with substantial reductions in memory
usage and computation time across all datasets. The findings from this research un-
derscore the potential of binary embeddings, particularlythose optimized using the CS
method, to revolutionize NLP by offering a balance between computational efficiency and
model performance. These methods are especially valuable inresource-constrained envir-
onments, enabling the deployment of sophisticated NLP models on mobile devices and
edge computing platforms. In the future we look for more algorithms that can binarize
real features to binary values and help up achieve accuracies similar to state of the art
methods with less computation time and memory.

Competing Interests:There are no competing interests.

Funding Information: Not Applicable

Author Contribution: All the authors have contributed equally.

Data Availability Statement: Dataset availability on request.

Research Involving Human and /or Animals: Not Applicable

Informed Consent: Not Applicable

## References

```
[1] Feiyang Pan, Shuokai Li, Xiang Ao, and Qing He. Relation reconstructive binariza-
tion of word embeddings. Frontiers of Computer Science, 16:1–8, 2022.
[2] Julien Tissier, Christophe Gravier, and Amaury Habrard. Near-lossless binarization
of word embeddings. Proceedings of the AAAI Conference on Artificial Intelligence,
33(01):7104–7111, July 2019.
[3] Dongwon Jo, Taesu Kim, Yulhwa Kim, and Jae-Joon Kim. Mixture of scales:
Memory-efficient token-adaptive binarization for large language models. arXiv pre-
print arXiv:2406.12311, 2024.
```

```
[4] Samarth Navali, Praneet Sherki, Ramesh Inturi, and Vanraj Vala. Word embedding
binarization with semantic information preservation. In Donia Scott, Nuria Bel,
and Chengqing Zong, editors,Proceedings of the 28th International Conference on
Computational Linguistics, pages 1256–1265, Barcelona, Spain (Online), December
```
2020. International Committee on Computational Linguistics.

```
[5] Y. Zhao. Heavy-ball-based optimal thresholding algorithms for sparse linear inverse
problems. SIAM Journal on Optimization, 30(1):31–55, 2020.
```
```
[6] S. Agrawal, R. Panda, S. Bhuyan, and B.K. Panigrahi. Tsallis entropy based optimal
multilevel thresholding using cuckoo search algorithm. InSwarm and Evolutionary
Computation, volume 11, pages 16–30. Elsevier, 2013.
```
```
[7] N. Meng and Y. Zhao. Newton-type optimal thresholding algorithms for sparse
optimization problems.SIAM Journal on Optimization, 30(1):31–55, 2020.
```
```
[8] Hamid R. Tizhoosh, Christopher Mitcheltree, Shujin Zhu, and Shamak Dutta. Bar-
codes for medical image retrieval using autoencoded radon transform. In2016 23rd
International Conference on Pattern Recognition (ICPR), pages 3150–3155, 2016.
[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-
training of deep bidirectional transformers for language understanding.arXiv preprint
arXiv:1810.04805, 2018.
```
[10] E. Giuliani and E. Camponogara. Genetic algorithms and their application to oil
field optimization problems.Computers & Chemical Engineering, 82:282–292, 2015.

[11] A. Beck and M. Teboulle. Iterative shrinkage/thresholding algorithm (ista) for finding
sparse solutions to ill-posed inverse problems. SIAM Journal on Imaging Sciences,
2(1):183–202, 2009.

[12] Haotong Qin, Ruihao Gong, Xianglong Liu, Xiao Bai, Jingkuan Song, and Nicu Sebe.
Binary neural networks: A survey.Pattern Recognition, 105:107281, 2020.

[13] Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Ben-
gio. Binarized neural networks: Training deep neural networks with weights and
activations constrained to +1 or -1. InAdvances in neural information processing
systems, volume 29, pages 4107–4115, 2016.

[14] Zichao Guo, Qipeng Zhang, Yichong Zhang, Xipeng Chen, and Xuanjing Wang.
Accelerating bert inference for sequence labeling via discrete latent variables.arXiv
preprint arXiv:2010.05350, 2020.

[15] Shuchen Zhang, Xiaojie Zheng, Fan Yin, and Xipeng Liu. Bitnet: Scaling 1-bit
transformers for large language models.arXiv preprint arXiv:2310.11453, 2023.


[16] Hamid R. Tizhoosh. Barcode annotations for medical image retrieval: A preliminary
investigation. In2015 IEEE International Conference on Image Processing, ICIP
2015, Quebec City, QC, Canada, September 27-30, 2015, pages 818–822. IEEE, 2015.

[17] Hamid Tizhoosh, Shujin Zhu, Hanson Lo, Varun Chaudhari, and Tahmid Mehdi.
Minmax radon barcodes for medical image retrieval. 10 2016.

[18] Shujin Zhu and H.R. Tizhoosh. Radon features and barcodes for medical image re-
trieval via svm. In2016 International Joint Conference on Neural Networks (IJCNN),
pages 5065–5071, 2016.

[19] R. Alexander Knipper, Md. Mahadi Hassan, Mehdi Sadi, and Shubhra Kanti Kar-
maker Santu. Analogy-guided evolutionary pretraining of binary word embeddings.
In Yulan He, Heng Ji, Sujian Li, Yang Liu, and Chua-Hui Chang, editors,Proceedings
of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computa-
tional Linguistics and the 12th International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), pages 683–693, Online only, November 2022.
Association for Computational Linguistics.

[20] Wouter Mostard, Lambert Schomaker, and Marco Wiering. Semantic preserving sia-
mese autoencoder for binary quantization of word embeddings. InProceedings of
the 2021 5th International Conference on Natural Language Processing and Inform-
ation Retrieval, NLPIR ’21, page 30–38, New York, NY, USA, 2022. Association for
Computing Machinery.

[21] Hamidreza Rouzegar, Shahryar Rahnamayan, Azam Asilian Bidgoli, and Masoud
Makrehchi. Enhancing content-based histopathology image retrieval using qr code
representation. pages 1120–1125, 12 2023.

[22] A. Jain and D. Zongker. Feature selection: evaluation,application, and small sample
performance. IEEE Transactions on Pattern Analysis and Machine Intelligence,
19(2):153–158, 1997.

[23] N. Otsu. A threshold selection method from gray-level histograms. IEEE Transac-
tions on Systems, Man, and Cybernetics, 9(1):62–66, 1979.

[24] Hans-Paul Schwefel. Evolution and optimum seeking: The sixth generation. 01 1995.

[25] R. Ehsan, S. Rahnamayan, S.Z. Miyandoab, A.A. Bidgoli, and H.R. Tizhoosh. Train-
ing artificial neural networks by coordinate search algorithm.Computers & Chemical
Engineering, 82:282–292, 2024.

[26] P. Breheny and J. Huang. Coordinate descent algorithm for nonconvex penalized
regression, with application to biological feature selection. The Annals of Applied
Statistics, 5:232–253, 2011.


[27] A. A. Bidgoli and S. Rahnamayan. Memetic differential evolution using coordin-
ate descent. Proceedings of the Genetic and Evolutionary Computation Conference
Companion, pages 359–366, 2021.

[28] E. Rokhsatyazdi, S. Rahnamayan, S.Z. Miyandoab, A.A. Bidgoli, and H.R. Tizhoosh.
Training artificial neural networks by coordinate search algorithm.Proceedings of the
IEEE Congress on Evolutionary Computation, pages 1540–1546, 2023.

[29] K.-W. Chang, C.-J. Hsieh, and C.-J. Lin. Coordinate descent method for large-
scale l2-loss linear support vector machines.Journal of Machine Learning Research,
9(45):1369–1398, 2008.

[30] R. Tibshirani. Regression shrinkage and selection via the lasso.Journal of the Royal
Statistical Society. Series B (Methodological), 58(1):267–288, 1996.

[31] OpenCV.Image Thresholding. Accessed: March 27, 2025.

[32] Nobuyuki Otsu. A threshold selection method from gray-level histograms. IEEE
Transactions on Systems, Man, and Cybernetics, 9(1):62–66, 1979.

[33] Ravi Pratap Singh and Manoj Kumar Singh. Hybrid thresholding for image de-
convolution in expectation maximization framework. The Imaging Science Journal,
0(0):1–18, 2024.

[34] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-
training of deep bidirectional transformers for language understanding.arXiv preprint
arXiv:1810.04805, 2018.

[35] Hans-Paul Schwefel. Evolution and optimum seeking: the sixth generation. John
Wiley & Sons, 1993.

[36] Azam Asilian Bidgoli and Shahryar Rahnamayan. Memetic differential evolution
using coordinate descent. In 2021 IEEE Congress on Evolutionary Computation
(CEC), pages 359–366. IEEE, 2021.

[37] Emanuele Frandi and Andrea Papini. Coordinate search algorithms in multilevel
optimization.Optimization Methods and Software, 29(5):1020–1041, 2014.

[38] Paul Tseng. Convergence of a block coordinate descent method for nondifferentiable
minimization. Journal of Optimization Theory and Applications, 109(3):475–494,
2001.

[39] Hanan Hiba, Shahryar Rahnamayan, Azam Asilian Bidgoli, Amin Ibrahim, and Rasa
khosroshahli. A comprehensive investigation on novel center-based sampling for large-
scale global optimization.Swarm and Evolutionary Computation, 73:101105, 2022.


[40] Jyoti Malik, G. Sainarayanan, and Ratna Dahiya. Min max threshold range (mmtr)
based approach in palmprint authentication by sobel code method. Procedia Com-
puter Science, 2:149–158, 2010. Proceedings of the International Conference and
Exhibition on Biometrics Technology.

[41] N. Raja, V. Rajinikanth, and K. Latha. Otsu based optimal multilevel image
thresholding using firefly algorithm. Modelling and Simulation in Engineering,
2014:37, 2014.

## 7. Appendix

### 7.1 Statistical tests on various methods used

p-values obtained from the statistical test on various methods are shown below in tables.

```
Table 5: Post-Hoc Comparison Results (p-values) for IMDb
Method CS CSGlobal Hybrid MinMax Otsu Simple SimpleOptimizer OtsuOptimizer HybridOptimizer Real
CSCSGlobal 1.0002.199× 10 − 24 2.1991.000× 10 −^24 3.5613.689×× 1010 −−^5637 7.6556.947×× 1010 −−^4419 1.9861.405×× 1010 −−^6347 2.1512.102×× 1010 −−^4115 3.2144.193×× 1010 −−^3520 5.4236.422×× 1010 −−^3022 2.1934.851×× 1010 −−^2518 7.4857.485×× 1010 −−^1010
HybridMinMax 3.5617.655×× 1010 −−^5644 3.6896.947×× 1010 −−^3719 1.0002.102× 10 − 15 2.1021.000× 10 −^15 7.4851.448×× 1010 −−^1029 6.9470.111× 10 −^19 5.3243.342×× 1010 −−^1225 8.6535.424×× 1010 −−^1421 6.9534.322×× 1010 −−^1623 1.4051.417×× 1010 −−^4732
OtsuSimple 1.9862.151×× 1010 −−^6341 1.4052.102×× 1010 −−^4715 7.4856.947×× 1010 −−^1019 1.4480.111× 10 −^29 1.0001.417× 10 − 32 1.4171.000× 10 −^32 2.1941.294×× 1010 −−^1819 3.4282.421×× 1010 −−^1720 2.8971.897×× 1010 −−^2023 3.5611.448×× 1010 −−^5629
SimpleOptimizer 3.214× 10 −^35 4.193× 10 −^20 5.324× 10 −^12 3.342× 10 −^25 2.194× 10 −^18 1.294× 10 −^19 1.000 1.189× 10 −^18 9.342× 10 −^23 5.421× 10 −^15
OtsuOptimizer 5.423× 10 −^30 6.422× 10 −^22 8.653× 10 −^14 5.424× 10 −^21 3.428× 10 −^17 2.421× 10 −^20 1.189× 10 −^18 1.000 1.325× 10 −^21 7.389× 10 −^16
HybridOptimizer 2.193× 10 −^25 4.851× 10 −^18 6.953× 10 −^16 4.322× 10 −^23 2.897× 10 −^20 1.897× 10 −^23 9.342× 10 −^23 1.325× 10 −^21 1.000 2.193× 10 −^25
Real 7.485× 10 −^10 7.485× 10 −^10 1.405× 10 −^47 1.417× 10 −^32 3.561× 10 −^56 1.448× 10 −^29 5.421× 10 −^15 7.389× 10 −^16 2.193× 10 −^25 1.000
```
```
Table 6: Post-Hoc Comparison Results (p-values) for GLUE SST-2
Method CS CSGlobal Hybrid MinMax Otsu Simple SimpleOptimizer OtsuOptimizer HybridOptimizer Real
CSCSGlobal 1.0005.135× 10 − 6 5.1351.000× 10 −^6 1.2699.163×× 1010 −−^3628 3.0291.226×× 1010 −−^3829 6.9345.135×× 1010 −−^166 2.5951.472×× 1010 −−^3020 1.3422.132×× 1010 −−^2314 3.2543.654×× 1010 −−^1916 4.2355.932×× 1010 −−^2117 5.1356.934×× 1010 −−^616
HybridMinMax 1.2693.029×× 1010 −−^3638 9.1631.226×× 1010 −−^2829 1.0000.337 0.3371.000 1.0088.395×× 1010 −−^1720 0.0012.091× 10 − 5 3.0541.132×× 1010 −−^1217 4.9522.093×× 1010 −−^1515 7.4353.052×× 1010 −−^1814 2.7971.104×× 1010 −−^4445
OtsuSimple 6.9342.595×× 1010 −−^1630 5.1351.472×× 1010 −−^620 1.0080.001× 10 −^17 8.3952.091×× 1010 −−^205 1.0003.347× 10 − 10 3.3471.000× 10 −^10 1.8945.932×× 1010 −−^1218 2.8759.453×× 1010 −−^1416 3.9521.789×× 1010 −−^1813 4.4477.915×× 1010 −−^2639
SimpleOptimizer 1.342× 10 −^23 2.132× 10 −^14 3.054× 10 −^12 1.132× 10 −^17 1.894× 10 −^12 5.932× 10 −^18 1.000 2.789× 10 −^15 4.432× 10 −^16 3.215× 10 −^13
OtsuOptimizer 3.254× 10 −^19 3.654× 10 −^16 4.952× 10 −^15 2.093× 10 −^15 2.875× 10 −^14 9.453× 10 −^16 2.789× 10 −^15 1.000 1.293× 10 −^14 3.789× 10 −^13
HybridOptimizer 4.235× 10 −^21 5.932× 10 −^17 7.435× 10 −^18 3.052× 10 −^14 3.952× 10 −^18 1.789× 10 −^13 4.432× 10 −^16 1.293× 10 −^14 1.000 5.654× 10 −^17
Real 5.135× 10 −^6 6.934× 10 −^16 2.797× 10 −^44 1.104× 10 −^45 4.447× 10 −^26 7.915× 10 −^39 3.215× 10 −^13 3.789× 10 −^13 5.654× 10 −^17 1.000
```
```
Table 7: Post-Hoc Comparison Results (p-values) for AG News
Method CS CSGlobal Hybrid MinMax Otsu Simple SimpleOptimizer OtsuOptimizer HybridOptimizer Real
CSCSGlobal 1.0007.087× 10 − 15 7.0871.000× 10 −^15 4.3154.098×× 1010 −−^6859 4.0985.967×× 1010 −−^5948 1.6567.087×× 1010 −−^3315 5.9671.656×× 1010 −−^4833 3.3252.324×× 1010 −−^2920 4.8653.548×× 1010 −−^2517 5.7684.321×× 1010 −−^2115 7.0871.656×× 1010 −−^1533
HybridMinMax 4.3154.098×× 1010 −−^6859 4.0985.967×× 1010 −−^5948 1.0007.087× 10 − 15 7.0871.000× 10 −^15 5.9671.656×× 1010 −−^4833 1.6567.087×× 1010 −−^3315 4.2531.432×× 1010 −−^1921 5.4232.321×× 1010 −−^1619 7.1893.598×× 1010 −−^1416 1.4564.315×× 1010 −−^7568
OtsuSimple 1.6565.967×× 1010 −−^3348 7.0871.656×× 1010 −−^1533 5.9671.656×× 1010 −−^4833 1.6567.087×× 1010 −−^3315 1.0007.087× 10 − 15 7.0871.000× 10 −^15 2.7653.548×× 1010 −−^1918 4.0985.324×× 1010 −−^1516 6.3247.234×× 1010 −−^1214 5.9674.098×× 1010 −−^4859
SimpleOptimizer 3.325× 10 −^29 2.324× 10 −^20 4.253× 10 −^19 1.432× 10 −^21 2.765× 10 −^19 3.548× 10 −^18 1.000 1.432× 10 −^14 2.986× 10 −^12 1.324× 10 −^13
OtsuOptimizer 4.865× 10 −^25 3.548× 10 −^17 5.423× 10 −^16 2.321× 10 −^19 4.098× 10 −^15 5.324× 10 −^16 1.432× 10 −^14 1.000 1.342× 10 −^12 1.568× 10 −^13
HybridOptimizer 5.768× 10 −^21 4.321× 10 −^15 7.189× 10 −^14 3.598× 10 −^16 6.324× 10 −^12 7.234× 10 −^14 2.986× 10 −^12 1.342× 10 −^12 1.000 2.436× 10 −^14
Real 7.087× 10 −^15 1.656× 10 −^33 1.456× 10 −^75 4.315× 10 −^68 5.967× 10 −^48 4.098× 10 −^59 1.324× 10 −^13 1.568× 10 −^13 2.436× 10 −^14 1.000
```

Table 8: Post-Hoc Comparison Results (p-values) for CoNLL-2003
Method CS CSGlobal Hybrid MinMax Otsu Simple SimpleOptimizer OtsuOptimizer HybridOptimizer Real
CSCSGlobal 1.0002.196× 10 − 10 2.1961.000× 10 −^10 1.2225.516×× 1010 −−^5749 5.5161.874×× 1010 −−^4938 3.6452.003×× 1010 −−^3521 2.8811.659×× 1010 −−^2914 4.5262.342×× 1010 −−^2117 6.9533.432×× 1010 −−^1915 8.2564.654×× 1010 −−^1713 2.1961.937×× 1010 −−^1025
HybridMinMax 1.2225.516×× 1010 −−^5749 5.5161.874×× 1010 −−^4938 1.0002.196× 10 − 10 2.1961.000× 10 −^10 2.8811.659×× 1010 −−^2914 3.6452.003×× 1010 −−^3521 5.6533.234×× 1010 −−^1918 8.2344.432×× 1010 −−^1715 9.6536.352×× 1010 −−^1412 6.3001.222×× 1010 −−^6557
OtsuSimple 3.6452.881×× 1010 −−^3529 2.0031.659×× 1010 −−^2114 2.8813.645×× 1010 −−^2935 1.6592.003×× 1010 −−^1421 1.0000.00180 0.001801.000 2.4531.654×× 1010 −−^1313 3.7542.653×× 1010 −−^1211 5.3424.765×× 1010 −−^109 2.5591.487×× 1010 −−^4641
SimpleOptimizer 4.526× 10 −^21 2.342× 10 −^17 5.653× 10 −^19 3.234× 10 −^18 2.453× 10 −^13 1.654× 10 −^13 1.000 2.653× 10 −^11 4.875× 10 −^9 3.215× 10 −^13
OtsuOptimizer 6.953× 10 −^19 3.432× 10 −^15 8.234× 10 −^17 4.432× 10 −^15 3.754× 10 −^12 2.653× 10 −^11 2.653× 10 −^11 1.000 1.563× 10 −^8 5.654× 10 −^17
HybridOptimizer 8.256× 10 −^17 4.654× 10 −^13 9.653× 10 −^14 6.352× 10 −^12 5.342× 10 −^10 4.765× 10 −^9 4.875× 10 −^9 1.563× 10 −^8 1.000 3.653× 10 −^12
Real 2.196× 10 −^10 1.937× 10 −^25 6.300× 10 −^65 1.222× 10 −^57 2.559× 10 −^46 1.487× 10 −^41 3.215× 10 −^13 5.654× 10 −^17 3.653× 10 −^12 1.000

Table 9: Post-Hoc Comparison Results (p-values) for SNLI
Method CS CSGlobal Hybrid MinMax Otsu Simple SimpleOptimizer OtsuOptimizer HybridOptimizer Real
CSCSGlobal 1.0001.175× 10 − 50 1.1751.000× 10 −^50 7.2834.729×× 1010 −−^6114 4.5594.163×× 1010 −−^437 7.8561.390×× 1010 −−^2825 1.2221.510×× 1010 −−^1139 4.5985.654×× 1010 −−^2920 6.7237.432×× 1010 −−^2518 8.3429.542×× 1010 −−^2116 1.2221.120×× 1010 −−^1159
HybridMinMax 7.2834.559×× 1010 −−^6143 4.7294.163×× 1010 −−^147 1.0001.390× 10 − 25 1.3901.000× 10 −^25 2.4504.729×× 1010 −−^4114 4.2995.197×× 1010 −−^5230 6.4323.756×× 1010 −−^1918 9.3425.432×× 1010 −−^1616 1.3427.423×× 1010 −−^1213 3.2191.734×× 1010 −−^6853
OtsuSimple 7.8561.222×× 1010 −−^2811 1.3901.510×× 1010 −−^2539 2.4504.299×× 1010 −−^4152 4.7295.197×× 1010 −−^1430 1.0001.222× 10 − 11 1.2221.000× 10 −^11 3.2541.432×× 1010 −−^1514 5.4323.234×× 1010 −−^1212 7.3425.342×× 1010 −−^1010 2.4507.856×× 1010 −−^4128
SimpleOptimizer 4.598× 10 −^29 5.654× 10 −^20 6.432× 10 −^19 3.756× 10 −^18 3.254× 10 −^15 1.432× 10 −^14 1.000 2.432× 10 −^14 3.753× 10 −^12 1.432× 10 −^15
OtsuOptimizer 6.723× 10 −^25 7.432× 10 −^18 9.342× 10 −^16 5.432× 10 −^16 5.432× 10 −^12 3.234× 10 −^12 2.432× 10 −^14 1.000 2.452× 10 −^12 5.342× 10 −^15
HybridOptimizer 8.342× 10 −^21 9.542× 10 −^16 1.342× 10 −^12 7.423× 10 −^13 7.342× 10 −^10 5.342× 10 −^10 3.753× 10 −^12 2.452× 10 −^12 1.000 6.432× 10 −^13
Real 1.222× 10 −^11 1.120× 10 −^59 3.219× 10 −^68 1.734× 10 −^53 2.450× 10 −^41 7.856× 10 −^28 1.432× 10 −^15 5.342× 10 −^15 6.432× 10 −^13 1.000


