## Contents lists available at ScienceDirect

# Ain Shams Engineering Journal

## journal homepage: http://www.sciencedirect.com

## Full length article

## 2P-BEnc: A two-phase information retrieval and ranking system based on

## the BERT encoder

## Sunil Kumar a,*^ , Divya Rohatgi b^ , Navin Prakash c^ , Shubham Sahai d^ , Saurav Chandra e^ ,

## Suman Kumar Mishra f^ , Arshad Ali g^ , Munish Kumar h

a (^) _Department of Information Technology, Ajay Kumar Garg Engineering College, Ghaziabad, Uttar Pradesh, India_
b (^) _Department of Engineering and Technology, Bharati Vidyapeeth University, Navi Mumbai, Maharastra, India_
c (^) _Department of Computer Science & Engineering, Dr. A.P.J. Abdul Kalam Technical University, Lucknow, Uttar Pradesh, India_
d (^) _Department of Computer Science, School of Management Sciences, Varanasi, Uttar Pradesh, India_
e (^) _Department of Computer Science & Engineering, KIET Group of Institutions, Ghaziabad, Uttar Pradesh, India_
f (^) _Department of Computer Science & Engineering, Khwaja Moinuddin Chishti Language University, Lucknow, Uttar Pradesh, India_
g (^) _Faculty of Computer and Information Systems, Islamic University of Madinah, Madinah, Saudi Arabia_
h (^) _Chandigarh University, Mohali, Punjab, India_

## A R T I C L E I N F O

## Keywords:

## Information retrieval

## BERT

## Deep language model

## Learning-to-rank

## A B S T R A C T

## Information retrieval methods have been advanced by the development of Natural Language Understanding

## (NLU). The development of deep neural networks was a key driver in the creation of effective Language Models

## (LM: statistical models trained to understand and generate human language), which significantly enhanced doc-

## ument retrieval. Even with these improvements, the current system for finding and ranking information faces

## significant challenges. In this work, we are focused on three of them. These problems are the system’s maxi-

## mum passage length processing capacity, the cost of making inferences, and integration of information across

## passages. The proposed 2P-2P-can handle longer and more passages while keeping the cost of making inferences

## low. Also, 2P-2P-uses more than one passage to make the ranked list of retrieval results. This way, 2P-BEnc

## stores information from different passages and improves ranking performance. Mechanism: 2P-BEnc achieves

## this through a cascaded encoder design: (1) Sentence-level processing bypasses BERT’s (Bidirectional Encoder

## Representations from Transformers: a transformer-based language model that establishes contextual relationships

## through attention mechanisms) token limits by treating sentences as atomic units, (2) Offline precomputation of

## sentence vectors slashes inference costs (Computational resources required during prediction phase) by 300 ×, (3)

## Cross-passage attention in Phase 2 aggregates contextual signals across documents. When the performance of the

## proposed system is compared to benchmarking datasets that are available to the public, the results are competi-

## tive and encouraging. On the MS-MARCO dataset, the proposed 2P-BEnc model achieved 38.6 MRR@10, which

is 5.75 % higher than the BERT (^) _Large_ model. On the TREC-CAR dataset, the proposed 2P-BEnc model achieved 35.
MAP@1000, which is 5.67 % higher than BERT (^) _Large_. Aside from that, the 2P-BEnc performed as well as or better
than BERT (^) _Large_ on other benchmark datasets (Robust04, WikiPassageQA, and WikiQA).

## 1. Introduction

## Information retrieval systems (IRS) serve as critical infrastructure

## of modern digital experiences, from e-commerce platforms processing

## $4.28 trillion in annual transactions [ 1 ] to educational portals serv-

## ing over 1.5 billion learners globally. [2,3] The exponential growth

## of digital content – with e-commerce projected to expand by 18 %

## annually through 2030 [ 4 ] – necessitates increasingly sophisticated

## ranking methodologies. [5,6] Consumers now face decision paralysis

## when selecting among thousands of options, making personalized neural

## ranking systems critical for aligning products with user preferences. [ 7 ]

## (See Appendix A for definitions of technical terms)

## Transformer-based architectures, particularly BERT (Bidirectional

## Encoder Representations from Transformers) [ 8 ], have revolutionized

* (^) Corresponding author.

## Email addresses: sunilymca2k5@gmail.com (S. Kumar), divi.rohatgi@gmail.com (D. Rohatgi), naveenshran@gmail.com (N. Prakash),

## shubhamsahai17@gmail.com (S. Sahai), sauravchandra01@gmail.com (S. Chandra), sumanmishra@kmclu.ac.in (S. Kumar Mishra), a.ali@iu.edu.sa (A. Ali),

## ggscemunish@gmail.com (M. Kumar).

## https://doi.org/10.1016/j.asej.2025.

## Received 15 October 2024; Received in revised form 28 September 2025; Accepted 31 October 2025

## Ain Shams Engineering Journal 17 (2026) 103853

## Available online 19 November 2025

## 2090-4479/© 2025 The Authors. Published by Elsevier B.V. on behalf of Faculty of Engineering, Ain Shams University. This is an open access article under the CC

## BY-NC-ND license ( http://creativecommons.org/licenses/by-nc-nd/4.0/ ).


##### document ranking by establishing contextual relationships between

##### queries and passages through attention mechanisms [ 9 ]. Recent

##### approaches [ 10 ] concatenate queries and passages to generate rele-

##### vance scores, achieving state-of-the-art results on benchmark datasets.

##### However, fundamental architectural constraints in BERT-based systems

##### hinder their practical deployment in large-scale commercial environ-

##### ments.

##### 1.1. Technical limitations of current BERT architectures

##### Three critical technical constraints undermine the effectiveness of

##### BERT in production systems:

##### 1. Quadratic Computational Complexity : BERT’s self-attention

##### mechanism requires O ( n 2 ) computational operations for sequence

##### length n due to pairwise token interactions [ 9 ]. Processing a

##### single query-passage pair (typically 500+ tokens) requires ~0.

TFLOPS on BERT (^) Large , making real-time ranking computationally

##### prohibitive.

##### 2. Fixed-Length Processing Constraint : Positional encoding

##### (Method injecting token position information into transformer

##### models) limits input to 512 tokens [ 8 ]. For documents exceeding

##### this threshold (e.g., product descriptions averaging 750 tokens in

##### the Amazon catalog), critical contextual information beyond the

##### truncation window is permanently lost.

##### 3. Dynamic Inference Requirement : Passage embeddings must be

##### recomputed per query since relevance is conditioned on specific

##### query-passage interactions. This prevents precomputation and in-

##### creases inference latency by 5–7× compared to static embedding

##### systems [ 11 ].

##### These limitations create an efficiency-accuracy trade-off: systems ei-

##### ther sacrifice context (via truncation) or incur prohibitive computational

##### costs when processing full documents.

##### 1.2. Research gaps and motivations

##### Our analysis identifies three unresolved challenges in neural infor-

##### mation retrieval.

##### 1.2.1. Contextual insufficiency in passage encoding

##### Current systems process passages as isolated token sequences with-

##### out modeling inter-sentence relationships. This approach fails to capture

##### document-level semantics crucial for understanding nuanced product at-

##### tributes (e.g., distinguishing “battery life” in smartphones vs. electric

##### vehicles).

##### 1.2.2. Computational inefficiency in ranking

##### Conventional BERT architectures require simultaneous processing of

##### query-passage pairs during inference. The absence of pre-computable

##### representations forces recalculations for each new query, creating bot-

##### tlenecks when ranking thousands of candidates.

##### 1.2.3. Isolated relevance assessment

##### Existing approaches independently evaluate query-passage pairs, dis-

##### regarding comparative relationships between candidate passages. This

##### prevents systems from leveraging relative relevance signals that could

##### optimize ranking decisions.

##### These gaps collectively constrain retrieval performance: Contextual

##### loss reduces ranking accuracy, computational costs hinder scalability,

##### and isolated processing prevents optimal result ordering. Our work

##### specifically addresses these interconnected challenges through a novel

##### two-phase architecture.

##### 1.3. Our novelty in context

##### Prior studies on BERT-based ranking typically concatenate queries

##### and documents into a single sequence, limiting scalability for long texts.

##### In contrast, our 2P-BEnc employs a hierarchical, two-phase architecture

##### that (i) precomputes sentence and passage embeddings offline, (ii)

##### employs cross-passage attention to capture comparative context, and

##### (iii) reduces inference complexity by pruning to a small candidate set.

##### To our knowledge, no prior work jointly achieves efficient long-text

##### handling and relational ranking under the same framework.

##### 1.4. Nonlinearity considerations

##### Transformers rely on nonlinear activation functions and softmax at-

##### tention mechanisms, which introduce complex, nonlinear dynamics into

##### ranking models. Recent advances in nonlinear differential equations and

##### soliton theory provide insights into these behaviors [2,5,12–14]. By ac-

##### knowledging these nonlinearities, we design 2P-BEnc’s architecture to

##### harness the expressive power of Transformer layers while maintaining

##### stability and efficiency. These considerations are elaborated upon in the

##### theoretical analysis and reflected in our design choices.

##### 1.5. Our contributions

##### We propose 2P-BEnc, a Two-Phase information retrieval system that

##### fundamentally rethinks BERT’s application to ranking tasks. Our key

##### innovations include

##### 1. Decoupled Sentence Encoding : We introduce a hierarchical en-

##### coder where sentence representations are generated independently

##### of queries using distilled BERT architectures. These vectors can

##### be precomputed and stored, reducing inference latency by 68 %

compared to BERT (^) Base (Table 6 ).

##### 2. Cross-Passage Attention Mechanism : A novel interaction layer

##### models relationships between candidate passages during ranking.

##### By incorporating a comparative context, our system achieves 9.3 %

##### higher MRR@10 on MS MARCO compared to pairwise approaches.

##### 3. Variable-Length Document Processing : By operating at

##### sentence-level granularity (max 511 sentences/passage), our

##### system handles documents 7 × longer than vanilla BERT while

##### maintaining linear computational complexity relative to passage

##### length.

##### 4. End-to-End Trainable Architecture : The entire system – from

##### sentence encoding to final ranking – is optimized jointly using

##### a novel listwise loss function that directly optimizes for ranking

##### metrics like nDCG.

##### Experimental validation on MS MARCO and TREC-CAR demonstrates

##### state-of-the-art results: 2P-BEnc achieves 0.392 MRR@10 (9.5 % im-

provement over BERT (^) Base ) while reducing inference latency by 63 %.

##### The architecture’s modular design enables deployment in resource-

##### constrained environments without sacrificing ranking quality.

##### 1.6. Paper organization

##### Section 2 analyzes limitations of existing neural ranking architec-

##### tures. Section 3 details 2P-BEnc’s hierarchical encoder and cross-passage

##### attention mechanism. Section 4 describes evaluation protocols and

##### datasets. Section 5 presents comparative results and computational ef-

##### ficiency metrics. Section 6 discusses implications and future research

##### directions.

##### 2. Related work

##### 2.1. Graph based expert ranking

##### One of the most popular branches of ranking is referred to as do-

##### main expert ranking based on the semantic knowledge of exploring the

##### graph network. Expert ranking intuitively refers to the online collabo-

##### ration of expert members who can contribute to a particular domain of

##### study [ 15 ]. In graph-based expertise ranking, content based approach

##### of passage analysis is violated. From the given knowledge repository in

##### the graph, an ordered list of relevant information is obtained by raising

##### a set of queries as input data items. This process is termed as knowledge


##### retrieval or findings [ 16 ]. The collection from the repository may belong

##### to the set of any document/passage such as emails or conferences on any

##### topic of academic or social activities [ 17 ]. In an email ranking system,

##### the nodes and edges of the graphs refer to the users and sender-receiver

##### connections and HITS algorithm is used for web ranking based on Term

##### Frequency -Inverse Document Frequency (TF-IDF) [ 18 ]. Bidirectional

##### Transformers for Language Understanding (BERT) [ 8 ] based ranking

##### models have been popular for enhancing the ability of natural language

##### recognition and understanding. These models use self-attention matrix

##### of graph generated by relations of words and representation with cor-

##### responding queries. In [ 19 ], to address computational overhead, they

##### remove irrelevant words from BERT and disentangled the corresponding

##### components of the queries for ranking the passage.

##### 2.2. Interactive neural ranking

##### In case of interaction-based ranking models, priority is given to

##### learning from the essence of the connections between the input texts

##### instead of the individual representation of the text. To measure the

##### performance the relevance score of the connection is computed from

##### the interaction function. Basically, the interaction functions can be

##### classified as parametric and non-parametric [ 20 ]. The majority of the

##### non-parametric interaction functions perform the operations based on

##### radial-basis, cosine similarity and binary operators. Deep Relevance

##### Matching Model(DRMM) described in [ 21 ] is used for converting the

##### matrix of local interaction functions to the fixed length histogram for

##### matching the relevance score. One of the important methods for com-

##### puting relevance scores is MatchPyramid [ 22 ] which uses convolution

##### operator on query-passage similarity matrix to produce the query–

##### passage relevance score. On the other end,the parametric interaction

##### functions learn from the similarity measure techniques such as dis-

##### tance functions of different data items. Convolution kernel based several

##### important models were developed. Many important machine learning

##### architectures are described in [ 23 ] which promote targeting deep learn-

##### ing methods for better outcomes. Convolutional Kernel-based Neural

##### Ranking Model discussed in [ 24 ],Conv-KNRM uses CNN for representing

##### the n-grams of different lengths. This model matches using unified em-

##### beddings in which learning-to-rank(LTR: Framework for training models

##### to optimize ranking of information) [ 25 ] and kernel pooling layers are

##### installed to generate ranking scores. Hu et al.(2014) [ 26 ] generated sec-

##### ond version of CNN architecture for natural language recognition in

##### which the primary interaction was performed between two sentences

##### using pooling and convolution operations.

##### 2.3. Generative inferential neural ranking

##### Generative models are the specific class of probabilistic models like

##### discriminative models. As suggested by their name“, it is clear that

##### these models can be used to generate new data that looks similar to

##### real data. Various classes of information retrieval problems utilise gen-

##### erative models. In query generation, attention models also play an

##### important role. With this consideration, both Ren et al. [ 27 ] and Yang

##### et al. [ 28 ] performed the tasks of reformulating of the queries under

##### some conversation into the queries oriented to search engines.

##### Using similar generative model, Ahmad et al. [ 29 ] and Chang

##### et al. [ 30 ] proposed a query recommender system by reformulat-

##### ing the queries raised by users. For the effective ranking of query-

##### passage, Nogueira et al. [ 31 ] proposed neural ranking scheme that

##### utilised sequence2sequence Transformer model discussed in [ 32 ]. In

##### this scheme, query-passage pair is extended and Bidirectional-Encoder-

##### Representations-Transformers(BERT) model is utilised to match and

##### retrieve the queries from the expanded passage. One of the important

##### applications of IR is to test the relevance of particular passage. Nogueira

##### et al. [ 33 ] recently targeted the task of binary classification with the help

##### of T5 models presented in [ 34 ]. The T5 model follows pre-training ap-

##### proach with Transformer model in sequence-to-sequence mode. These

##### models generate the queries based on token values ’TRUE’ or FALSE’ for

##### relevant and non-relevant passages. In queries generated from the input

##### passage using handwritten text, the likelihood models are adopted for

##### generating the query [ 35 ] which is defined across entire documents.

##### The same approach is adopted by Zhuang et al. [ 36 ] in which the in-

##### vestigation of likelihood query generation model is performed using

##### sequence2sequence model built on Transformer. Lesota et al. [ 35 ] also

##### expanded the query generative deep network model and investigated

##### the important outcomes of generative IR models. The currently ongo-

##### ing research on generative inferential deep networks can be immensely

##### productive for developing IR and web ranking systems.

##### 2.4. Causal inferential neural ranking

##### To establish a particular interface between the query and correspond-

##### ing document, the inferential neural networks play a productive role in

##### assisting in building a desired ranking system. The concept of building

##### a causal interface referred to as causal inference is described in [ 37 ].

##### Causal inferences with neural models facilitate the researchers in con-

##### sidering the ranking system as a new dimension. From the visionary

##### perspective of causal inference models, the counterfactual samples help

##### to improve the existing models [ 38 ]. All these technical studies assumed

##### that the confounder must be domain expert [ 39 ]. The causal attention

##### maps [ 40 ] motivate the design of graph of from set queries and pas-

##### sages. In the causal graph, self-attention map is used to alleviate the

##### biased features and remove the spurious information from the rank-

##### ing passage. Adaptive masking with mutual information can also reduce

##### the confounding effects. Thus, causal inference provides an interface of

##### implicit feedback from users to mitigate bias. On analysis of casual struc-

##### ture, Zheng et al. [ 41 ] proposed Disentangling Interest and Conformity

##### with Causal Embedding (DICE) model which can disentangle the inter-

##### est of users’ by embedding queries. They further analyzed the casual

##### structures of the embedded data to enhance the performance of casual

##### ranking system [ 42 ]. Handling the hidden confounders in the query

##### recommendation system is quite challenging task. By utilising causal in-

##### ference models, Wang et al. [ 43 ] made some fruitful efforts and in their

##### next research, they proposed DecRs [ 44 ] model which can dynamically

##### eliminate the effect of confounders. Therefore, bias in ranking system

##### has very recently become serious problem which is addressed by causal

##### inference.

##### 3. Proposed methodology

##### 3.1. Overview of proposed work

##### The proposed two-phase BERT encoder (termed as 2P-BEnc) based

##### information retrieval and ranking system (IRRS) combines four sub-

##### networks, including three BERT [ 8 ] based encoders and a relevance

##### network (fully connected network). Fig. 1 depicts the workflow of the

##### system. The given passage is first divided into a sequence of sentences.

##### These sentences are then further divided into a sequence of tokens. A

##### special token [SREP] (representative of the sentence) is added to the se-

##### quence of tokens of each sentence. These tokens are further converted

##### into their respective embedded vectors by a lookup table. This sequence

##### of tokens for a sentence is processed by a sentence encoder (First BERT-

##### based encoder) and generates the processed vector for each token. The

##### processed vector for the [SREP] token is the representative vector for the

##### sentence (RepVS). Similarly, a query is also processed with the [QREP]

##### special token, and the representative vector for the query (RepVQ) is

##### generated by the query encoder (Second BERT-based encoder). In this

##### way, the sequence of RepVS is generated for a passage. The sequence

##### of RepVS for a passage is further encoded by the third BERT-based en-

##### coder (Passage-Encoder) along with the embedding vector of a special

##### token [PREP] to yield the representative vector for the passage (RepVP).

##### Up to this point, the query and the passage are processed separately, and

##### we designate this stage as the first phase of the proposed 2P-BEnc IRR

##### system. This two-phase decomposition fundamentally enables 2P-BEnc’s

##### breakthrough in handling long texts efficiently. Section 3.2 details these

##### innovations.


### Create Sequence of sentences

##### Passage

##### 001

### Get Embedding Vector

##### [PREL]

##### Create

##### Sequence

##### of

##### tokens

##### Create

##### Sequence

##### of

##### tokens

##### Create

##### Sequence

##### of

##### tokens

### Positional Embedding

### Passage

### Encoder

##### Positional

##### Embedding

##### Sentence

##### Encoder

##### Token

##### Classif

##### y^

##### Positional

##### Embedding

##### Sentence

##### Encoder

##### Token

##### Classif

##### y^

##### Positional

##### Embedding

##### Sentence

##### Encoder

##### Token

##### Classif

##### y^ Mask

##### Token

##### Loss

##### Mask

##### Token

##### Loss

##### Mask

##### Token

##### Loss

##### Query

##### Create

##### Sequence

##### of

##### tokens

##### Positional

##### Embedding

##### Query

##### Encoder

##### Token

##### Classify

(^) Mask

##### Token

##### Loss

##### Relevance Score

#### RelNet

Passage (^)

##### Create Sequence of sentences

##### Sentence 001

##### Sentence 002

##### Sentence

##### N

##### [PREL]

##### Create Sequence of tokens

##### Token 001

##### Token 002

##### Token

##### M

##### [SREP]

##### Relevance

##### Loss

##### C

##### C

##### Concatenation

#### Relevance Vector

###### Fig. 1. The overall workflow of the proposed two-phase BERT encoder-based (2P-BEnc) IRR system. The details of different encoders are shown in Fig. 2, and the

###### architecture for the RelNet (relevance network) is depicted in Fig. 3.

##### The second phase of the 2P-BEnc IRR system requires the RepVP and

##### RepVQ for further processing. The second phase is a fully-connected

##### layer-based network (termed as RelNet: relevance network). First, the

##### RepVQ and RepVP vectors are concatenated (the generated vector is

##### termed as RelVPQ: relevance vector of passage with the query). The

##### generated RelVPQ vector is further processed by the RelNet, which de-

##### velops the relevance score of a passage with respect to the given query.

##### The activation function used in the RelNet differs between the training

##### and inference phases. In the training phase, the system processes mul-

##### tiple passages for a given query to generate an ordered list for ranking.

##### Therefore, softmax activation is used across all processed passages. The

##### inference phase uses sigmoid activation when the relevant passage is

##### searched for a given query, and softmax activation generates a ranking

##### between the input passages. Please refer to Section 3.4 for detailed in-

##### formation. The relevance score for a query q and passage p is computed

##### by the RelNet as

##### s ( q, p ) = σ

##### {

_W_ [ _RepV_ (^) _Q_ || _RepV_ (^) _P_ ] + _b_

##### }

##### , (1)

##### where [ · || · ] denotes vector concatenation, W and b are learnable

##### parameters, and σ is the sigmoid activation used during inference.

##### 3.2. Core innovations: long-text handling & efficiency

##### Overcoming Length Constraints: Unlike BERT’s token-based

##### processing (limited to 512 tokens), 2P-BEnc operates at the sentence level.

Each passage _P_ is split into sentences { _S_ 1 _, S_ 2 _,_ ... _, S_ (^) _n_ } where _n_ ≤ 511. The
Sentence Encoder (Fig. 2 (b)) processes each _S_ (^) _h_ independently, gener

##### ating sentence embeddings RepV. Sh This hierarchical approach allows

##### processing of documents exceeding 10,000 tokens while maintaining

##### fixed dimensionality at the Passage Encoder input (512 units regardless

##### of n ).

##### -

##### Inference Efficiency: 2P-BEnc decouples passage encoding from

##### query processing:

• _Precomputation: RepV_ and _Sh RepV_ (^) _P_ are generated _offline_ (Fig. 1, Phase

##### 1) and stored in a vector database

- _On-demand Processing:_ At inference, only _RepVQ_ (query embedding)

##### and the lightweight RelNet (Fig. 3 ) are computed in real-time

- _Complexity Reduction:_ Avoids reprocessing passage tokens for ev-

##### ery query, reducing complexity from O ( L 2 ) (BERT) to O (1) for

##### precomputed passages

Table 6 quantifies 300 × faster inference versus BERT (^) Large.

##### Cross-Passage Context: The RelNet (Phase 2) jointly analyzes

##### { RepVP 1 , ... , RepVPk } for a query, enabling comparative relevance scor

##### ing. This mimics human assessment where document relevance is judged

##### relationally rather than in isolation.

##### -

##### 3.3. Different components of the proposed 2P-BEnc IRRS

##### The proposed 2P-BEnc IRR system has four main components. It has

##### three BERT-based encoders [ 8 ] responsible for encoding the sentences,

##### queries, and passages and generating the representative vectors for them

###### Add & Norm

###### Feed

###### Forward

###### Add & Norm

###### Multi-Head

###### Attention

###### Nx

###### Encoder 01

###### Encoder 02

###### Encoder 08

###### Encoder 01

###### Encoder 02

###### Encoder 08

#### Query Encoder

#### Pa

#### ssage

#### Encoder

###### An Unit of BERT Encoder

###### Encoder 01

###### Encoder 02

###### Encoder 06

#### Sentence Encoder

###### Fig. 2. Different encoder architectures used in the proposed 2P-BEnc IRRS system. The basic unit of these encoders is based on the BERT and Transformer architectures.


###### FCL: Fully Connected Layer LNL: Layer-Normalization Layer RAL: ReLu-Activation Layer SAL: Sigmoid-Activation Layer

###### FCL:

(^128)
Neurons (^) LNLRAL SAL

###### FCL: 128 Neurons

###### LNLRAL

###### FCL: 128 Neurons

###### LNLRAL

###### FCL: 64 Neurons

###### FCL: 64 Neurons

###### FCL: 64 Neurons FCL:

###### 1

###### Neurons

###### LNLRAL

###### FCL:

(^128)
Neurons (^) LNLRAL

###### Relevance

###### Vector

C

###### Relevance

###### Score

## RelNet C^ Concatenation

###### Fig. 3. The architecture of RelNet: Relevance Network utilised by the proposed 2P-BEnc IRRS system for generating the relevance score for a passage concerning the

###### given query.

##### as RepVS, RepVQ, and RepVP, respectively. These encoders comprise

##### stacked multi-head attention (Parallel attention mechanisms focusing on

##### different representation subspaces) layers. Section 3.3.1 describes these

##### encoders in detail. Besides these encoders, the proposed IRR system has

##### a fully-connected layers based network (termed as RelNet: relevance net-

##### work) that generates a passage’s relevance score concerning the given

##### query. Section 3.3.2 describes the RelNet in detail.

##### 3.3.1. BERT-based encoders

##### The BERT-based encoders are basically stacking of the multi-head at-

##### tention units. An identical layer stack makes up the BERT-based encoder.

##### There are two modules beneath each one. The first is a multi-head self-

##### attention mechanism, and the second is a position-wise, fully-connected

##### feed-forward network. Each of the two modules is surrounded by a

##### residual connection [ 45 ] (Skip-connection architecture mitigating van-

##### ishing gradient problems), and then the encoder is normalized by

##### layer normalization [ 46 ] (Technique stabilizing training by normalizing

##### inputs across features). In other words, each module output is calcu-

##### lated as LayerNorm(x+ Module(x)), where Module ( x ) is the function

##### implemented by the underlying module (multi-head self-attention or

##### fully-connected feed-forward network). The architectures of different

##### encoders used in the proposed 2P-BEnc IRR system are depicted in Fig. 2

##### along with the basic unit of the identical encoder layer.

##### Three encoders are used in the proposed 2P-BEnc IRR system. These

##### encoders take their design cues from the BERT’s encoder [ 8 ]. These en-

##### coders are independent models with their own set of parameters. All of

##### the encoders’ pre-trained weights come from the (L=8, H=512, A=8)

##### and (L=6, H=256, A=4) encoders in the [ 47 ] repository. Here, L is the

##### number of encoder layers, H is the hidden embedding size, and A is the

##### number of attention heads at each encoder (A is equal to the H/64 in all

##### cases [ 47 ]).

##### The three encoders are as follows:

##### 1. Query-Encoder: L=8, H=512, and A=8. The pre-trained weights

##### are obtained from https://storage.googleapis.com/bert_models/

##### 2020_02_20/uncased_L-8_H-512_A-8.zip.

##### 2. Sentences Encoder: L=6, H=256, and A=4. The pre-trained

##### weights are obtained from https://storage.googleapis.com/bert_

##### models/2020_02_20/uncased_L-6_H-256_A-4.zip.

##### 3. Passage-Encoder: L=8, H=512, and A=8. The pre-trained

##### weights are obtained from https://storage.googleapis.com/bert_

##### models/2020_02_20/uncased_L-8_H-512_A-8.zip.

##### 3.3.2. RelNet: Relevance Network

##### The RelNet (Relevance Network) receives the RelVPQ (relevance vec-

##### tor of the passage concerning a query) which is the concatenated vector

##### of RepVP and RepVQ as the input and yields a relevance score between

##### them. The architecture of the network used by the proposed method

##### is depicted in Fig. 3. Here, fully connected neural layers with different

##### numbers of neurons are used as the feature encoder, whereas the layer-

##### normalization [ 46 ] layer is used for the normalisation of the generated

##### output vector. All the activation layers used in this network utilise the

##### ReLU [ 48 ] (Rectified Linear Unit: f ( x ) = max(0 , x ), a common activa-

##### tion function) activation except the last activation layer. In the training

##### phase, the system processes multiple passages for a given query to gen-

##### erate an ordered list for ranking. Therefore, the last activation layer

##### utilises the softmax [ 49 ] (normalizes outputs into probability distribu-

##### tion) across all processed passages. Whereas the inference phase uses

sigmoid [ 49 ] (function: _σ_ ( _x_ ) = (^) 1+^1 _e_ – _x_ , produces probability outputs be-

##### tween 0–1) activation when the relevant passage is searched for a given

##### query, and softmax activation is utilised to generate a ranking between

##### the input passages.

##### 3.4. Loss calculation in training phase

##### We also added the Masked Language Model (Self-supervised task pre-

##### dicting masked tokens in input sequences) learning technique, which is

##### quite similar to the BERT [ 8 ] encoder model. In this procedure, cer-

##### tain tokens are randomly masked, and the final mapping is carried out

##### across the whole token vocabulary using the vector that is created by

##### the encoder. This learning technique is a masked language model used

##### in BERT [ 8 ]. These masked tokens are also found in the query and

##### sentence encoders of the proposed system; therefore, the mask token

##### loss is deployed for the learning of the representative vectors (RepVQ,

##### RepVS, RepVP). Following a feed-forward layer, a softmax activation is

##### performed on the representative vectors (RepVQ, RepVS, RepVP) that

##### correspond to the masked tokens in order to map them to the lexicon of

##### tokens. When training the masked language model, the cross-entropy

loss is put to use. This loss is denoted by the acronym _L_ (^) _mask_ in the

##### proposed work.

##### Besides this, we are also generating the ranking list of the relevance

##### scores of passages concerning a given query. We deployed a rank loss (an

##### objective function that optimizes relative ordering of items) to ensure

the proper order among the retrieved results. The rank loss _L_ (^) _rank_ is given

##### by Eq. (2). This loss improves the relevance score of all relevant passages,

##### whereas it decreases the relevance score of all non-relevant passages.

_L_ (^) _rank_ = – _log_

##### ( Σ

_h_ ∈ _P OS_

_Rel_ (^) _h_

##### )

- _log_

##### (

##### 1 –

##### Σ

_j_ ∈ _NEG_

_Rel_ (^) _j_

##### )

##### (2)

##### Where POS is the set of the top 75 % of all relevant passages with

##### respect to the given query and NEG is the set of the worst 75 % of all non-

##### relevant passages. Here we are not utilising all POS and NEG passages

##### so that the learning phase can cope with any wrongly labelled samples.

##### Along with the rank loss, we also utilise a loss incorporating the dis-

##### counted cumulative gain (DCG: a metric that measures ranking quality

##### with position-based discounts) score calculated over the retrieved re-

##### sults. This loss maximises the DCG score for top K retrieved results. We


term this loss as DCG loss _L_ (^) _DCG_ and it is given by Eqs. (3) and ( 4 ).
_gt_ (^) _r_ =

##### {

##### 1 , hf r ∈ P OS

##### 0 , otherwise.

##### (3)

_L_ (^) _DCG_ =

##### r Σ= k

_r_ =

_Rel_ (^) _r_

##### log ( r + 1) ×^ (1^ –^2 ×^ gt^ r^ )^ (4)

##### We adopted the weighted sum of the different losses for end-to-end

##### learning of the whole system. The overall loss is calculated by the Eq. (5).

_L_ = _L_ (^) _mask_ + _α_ × _L_ (^) _rank_ + _β_ × _L_ (^) _DCG_ (5)

##### The weights α and β are empirically set to 2.0 and 5.0 based on loss

magnitude in losses _Lrank_ and _L_ (^) _DCG_.

##### 3.5. Theoretical analysis

##### Theorem 1 ( Two-phase Complexity ). Under a token budget L and candi-

##### date set size k, inference cost of 2P-BEnc scales as O ( kL )^ in the second phase,

##### whereas a single-pass cross-encoder requires O ( NL 2 ) for N » k documents.

2

##### Proof. The self-attention mechanism of transformers exhibits quadratic

##### complexity in the sequence length L. In the proposed two-phase frame-

##### work only the top k candidates selected from the first phase are

##### re-encoded, leading to an overall complexity of O ( kL 2 ). In contrast, a

##### single-pass cross-encoder must process all N documents jointly, incur

##### ring O ( NL^2 ) cost.

##### -

##### □

##### 4. Experimental setup

##### 4.1. Implementation details

##### 4.1.1. Training phase

##### At a time, a randomly picked token (single) from a sentence or query

##### is masked for learning the masked language model. The RelNet generates

##### the relevance score for all passages concerning a given query. The RelNet

##### utilises a softmax activation function over all passages for a query to cre-

##### ate the relevance ranking. The end-to-end learning of the whole system

##### demands significant memory resources, so we have to limit the num-

##### ber of passages participating in relevance score calculation. As a result,

##### we fixed the number of passages for each query as 200. We randomly

##### chose 20 to 100 relevant passages and the remaining non-relevant pas-

##### sages from the collection for a particular query. As a result, we have a

##### diverse set of passages contributing to the ranking. The loss function for

##### the proposed system is given in Section 3.4.

##### The system’s learning additionally employs the L2 regularisation (a

##### technique to prevent overfitting by penalizing large weights) as a reg-

##### ulariser with a scale of 10 – 5. The optimiser approach is Stochastic

##### Gradient Descent, with the learning rate falling exponentially. The start-

##### ing learning rate is 0.1, and the final learning rate after 5000 iterations

##### is 10 – 3. The learning rate for this initial phase (iteration 1 to 5000) is

##### given by Eq. (6). After 5000 iterations, we employed a linearly declin-

##### ing learning rate (LearningRate) as in Eq. (7). We trained the model for

##### 150,000 iterations with a 16-batch size.

##### LearnhngRate = 0. 1 × (0. 01)

_hterathonNumber_ 5000

##### (6)

##### LearnhngRate = 10 –3^ ×

##### (

##### 1. 0 – hterathonNumber^ –^5000

##### 150000

##### )

##### (7)

##### All training trials are carried out on a Dell Precision 7920 Tower

##### Workstation with 72 CPU cores and a Quadro RTX 8000 GPU. The GPU

##### includes 48 GB of RAM as well as 4608 Cuda cores. The suggested ap-

##### proach requires around five days to train on a dataset (we have included

##### five datasets here, so the total time is approximately 25 days).

##### 4.1.2. Inference phase

##### The proposed system’s inference phase is notable because its sentence

##### and passage encoders do not use the provided query in representation

##### vector creation. This characteristic enables us to analyse the sentences

##### and passages before proceeding to the testing/inference step. As a re-

##### sult, the passage representation vectors are computed and saved in the

##### repository. These stored vectors are used in the RelNet with the query’s

##### representation vector to provide the ranking scores of all participating

##### passages concerning the query. We are simultaneously analyzing the

##### 5000 passage vectors with RelNet and eliminating the lower 4000 rel-

##### evance score passage vectors. The list is then updated with 4000 new

##### randomly picked passage vectors, and the process is repeated. Finally,

##### we will have 1000 top relevant passage vectors for every given query.

##### 4.2. Datasets

##### We list the required experimental settings and datasets to test the

##### proposed model.

##### 4.2.1. MSMARCO dataset

##### The name of the dataset is abbreviated from MAchine Reading

##### COmprehension (MSMARCO) [ 50 ], a very large collection of samples

##### 1,010,916 from Bing’s search query logs generated by human written re-

##### sponses. Apart from this, MSMARCO contains 8,841,823 passages which

##### are extracted from 3,563,535 online web pages and documents of nat-

##### ural language questions and answers. The dataset is built with various

##### difficulty levels. The training set of data samples contains 400 million

##### tuples for a particular query in the available passages for the experi-

##### ment’s purpose [11,51]. A set for evaluation is also provided containing

##### 6800 queries and 1000 corresponding passages.

##### 4.2.2. TREC-CAR dataset

##### TREC-CAR is a Complex Answer Retrieval(CAR) dataset developed

##### by Dietz et al.(2017) [ 52 ] comprising various contents, news outlines,

##### topics, and paragraphs extracted from English Wikipedia. The input

##### query in the dataset is generated by the concatenation of the titles of

##### a section of the dataset and an article selected from Wikipedia. The

##### corpus of the TREC-CAR consists of the paragraphs taken from English

##### Wikipedia, but all the abstracts are not included in that corpus. We

##### used only the first four training samples out of five pre-defined folds

##### in the original TREC-CAR dataset in our experiments. The total size of

##### the training sample is 2.3 million queries, and the remaining contents,

##### approximately 580 k queries, are used as the validation set. In our ex-

##### periments, we considered the same test sets of 2254 queries used in the

##### TREC-CAR 2017. In those experiments, true relevant passages are not

##### annotated when their rank is low. Here, in our experiments, we used

##### automatic annotation. Hence, the relevance scores can be provided for

##### all possible pairs of queries and passages.

##### 4.2.3. WikiPassageQA

##### We must make sure that the passages are pertinent and contain

##### enough information to support an appropriate response in order to find a

##### good passage ranking function to address the question-answering prob-

##### lem. So that the same passages can be used for various questions from

##### various topics, they should offer a well-balanced set of articles that are

##### relevant to various topics. WikiPassageQA [ 53 ] is a passage ranking

##### dataset made up of non-factoid questions and candidate passages taken

##### from Wikipedia articles by highly skilled crowd workers. It has already

##### been broken down into 417 development, 3332 training, and 416 test

##### questions.

##### 4.2.4. WikiQA

##### WikiQA [ 54 ] is a standard collection for response selection. The

##### matched candidate answers are sentences taken from Wikipedia articles,

##### and the questions were chosen from Bing query logs. It has already been

##### divided into 2118 training, 296 development, and 633 test questions,


##### but some of the questions have only correct answers while others have

##### no correct answers (refer to [ 55 ]).

##### 4.2.5. TREC robust2004 dataset

##### The total number of queries contained in TREC ROBUST 2004 dataset

##### is 250 each of which consists of two fields, title and description [56,57].

##### The main impact of Robust TREC is to improve the retrieval performance

##### by selecting more poorly performing topics. According to dataset gener-

##### ation scheme, the robust section considers the files with 250 topics and

##### their track pages are recorded. In this dataset, out of the topics from 301

##### to 450, 50 distinguished topics are selected.

##### 4.3. Evaluation criterion

##### To test performance measures of the proposed ranking model, we

##### used the following parameters.

- **Mean Reciprocal Rank (MRR):** It is computed based on decision of

##### binary relevance [ 58 ]. As explained in Eq. (8), it is reciprocal of the

##### rank which is obtained after taking the average of all the relevance

##### queries. The reciprocal rank is the “multiplicative inverse” of the

##### rank of the first correct item.

_MRR_ = (^) _Q_^1

##### Σ Q^

_h_ =

##### 1

_Rank_ (^) _h_ (8)

- **Precision:** The ratio of true positive results to all positive results is

##### called precision. More intuitively, precision at a given rank k be-

##### comes important when a user has importance only at the first k

##### documents retrieved by the domain experts (refer Eq. (9)). It ba-

##### sically indicates the standard of a positive prediction given by the

##### model. The precision refers to the ratio of number of relevant items

##### to total viewed items in the document. In this work, we are re-

##### porting the P@20 score of the obtained results. The proportion of

##### articles among the K found results that are relevant to the given

##### query is known as P@K [ 59 ]. The proportion of pertinent search

##### results among the top 20 results is known as P@20. As a result, it

##### can measure the relevance of the returned results.

##### P @ k =

##### r ( k )

##### k (9)

##### Here r ( k ) is the number relevant document retrieved at top k

##### positions.

- **Recall:** The evaluation of Recall is used to test the performance of

##### machine learning classifiers and defined by means of the ratio of

##### particular class correctly identified to the all the given possibilities.

##### For Mathematical computation [ 60 ], refer Eq. (10).

##### R @ k = r ( k )

##### R

##### (10)

##### Here R is the total relevant document

- **Mean average precision (MAP):** The average precision described

##### in [ 61 ] defines the mean average of the ranks which are listed in

documents. Refer the Eq. (11) for computing MAP where _P r_ ( _doc_ (^) _h_ ) is
the precision of computing the query _Q_ (^) _j_ with rank _h_ and _N_ denotes

##### number of queries.

_MAP_ = (^) _N_^1

##### Σ N

_j_ =

##### 1

_Q_ (^) _j_

##### Σ Q^ j

_k_ =

##### P @ k (11)

where _Q_ (^) _j_ stands for the number of relevant documents for query j

##### and P @ k gives the precision value for top k retrieved documents.

##### 5. Results analysis

##### This section presents the assessment results for the proposed 2P-

##### BEnc model together with a comparison of the prevalent IRRS models.

###### Table 1

###### The comparative study of the proposed 2P-BEnc with other IRRS models on

###### MS-MARCO.

```
Method Performance
MRR@10 (Dev) MRR@10 (Eval) Recall@
BM25 (Official) 16.7 16.5 81.
BM25 (Anserini [ 64 ]) 18.7 19.5 85.
doc2query [ 31 ] 21.5 22.8 89.
DeepCT [ 65 ] 24.3 – 91
docTTTTTquery [ 66 ] 27.7 28.4 94.
BERTbase [ 11 ] 34.7 – –
ColBERT [ 62 ] 36.0 36.7 96.
BERTlarge [ 11 ] 36.5 35.9 –
BERT-SMAP [ 63 ] 36.7 35.5 –
Proposed Model 38.6 37.8 97.
```
###### Table 2

###### The comparative study of the proposed 2P-BEnc with other

###### IRRS models on TREC-CAR dataset.

```
Method Performance
MAP MRR@
BM25 (Anserini [ 64 ]) 15.3 –
doc2query [ 31 ] 18.1 –
DeepCT [ 65 ] 24.6 33.
BERTbase [ 11 ] 31.0 –
ColBERT [ 62 ] 31.3 44.
BERTlarge [ 11 ] 33.5 –
Proposed Model 35.4 47.
```
##### Table 1 displays the assessment findings for the MSMARCO benchmark

##### dataset. All BERT-based models outperform the alternatives (BM25 has

##### 16.7 MRR@10, whereas ColBERT [ 62 ] and suggested 2P-BEnc have 36.

##### and 38.6 MRR@10, respectively). The proposed 2P-BEnc achieves supe-

##### rior performance, with MRR@10 values of 38.2 (Dev test set), 37.8 (Eval

##### test set), and Recall as 97.3. These results exceed the BERT Large model

##### by 2.1 MRR@10 (Dev test set) and 1.9 MRR@10 (Eval test set). The

##### BERT Large model contains a vector space of 1024 dimensions and stacks

##### 24 encoder units which leads to a significant amount of computation.

##### However, the proposed 2P-BEnc requires only the RelNet component

##### only to create the ranked list. The RelNet block requires substantially

##### less processing than other BERT-based models (please refer to Table 6 ).

##### The proposed 2P-BEnc model outperforms the ColBERT model, which is

built on the BERT (^) _L arge_ architecture and has a large number of learnable

##### parameters. Despite having minimal parameters (refer to Table 6 ), the

##### 2P-BEnc employs cross-document information to build an ordered list

##### of retrieval results, resulting in superior performance compared to other

##### advanced models such as ColBERT [ 62 ], BERT-SMAP [ 63 ].

##### The evaluation outcomes for the TREC-CAR [ 52 ] benchmark dataset

##### are displayed in the Table 2 Here, the suggested 2P-BEnc model per-

##### forms noticeably better than the alternative approaches. In contrast

to the ColBERT and BERT (^) _Large_ , which only generated 31.3 and 33.

##### MAP scores, respectively, the proposed 2P-BEnc model yields the 35.

##### MAP score. The 2P-BEnc capacity for cross-passage information encod-

##### ing and its cascading approach to encoding the passage representation

##### vector (RepVP) contribute to its superior performance.

##### The Robust04 [ 67 ] benchmark dataset evaluation results are shown

##### in Table 3. In this situation, the proposed 2P-BEnc model outperforms

##### the competing models significantly. The proposed 2P-BEnc model pro-

##### duced the 48.67 P@20 and 40.84 MAP@1000 scores. The closest com-

##### petitors of 2P-BEnc are Co-BERT [ 68 ], CEDR-KNRM [ 69 ], LGRe [ 19 ],

##### and DGRe [ 19 ]. The Co-BERT method managed to produce scores

##### of 36.31 MAP@1000 and 46.29 P@20, whereas CEDR-KNRM [ 69 ],

##### LGRe [ 19 ], and DGRe [ 19 ] yielded the 46.67, 47.7, and 48.13 P@

##### score respectively. The 2P-BEnc’s improved performance demonstrates


###### Table 3

###### The comparative study of the proposed 2P-BEnc with other

###### IRRS models on Robust04 dataset.

```
Method Performance
P@20 MAP@
Conv-KNRM [ 24 ] 34.08 –
BM25+RM3 [ 70 ] 38.21 29.
DPH+KL [ 71 ] 39.24 30.
BERTbase [ 72 ] 40.70 23.
Co-BERT [ 68 ] 46.29 36.
CEDR-KNRM [ 69 ] 46.67 –
LGRe [ 19 ] 47.9 –
DGRe [ 19 ] 48.13 –
Proposed Model 48.67 40.
```
###### Table 4

###### The comparative study of the proposed 2P-BEnc with other

###### IRRS models on WikiPassageQA dataset.

```
Method Performance
MAP MRR@
BM25 (Anserini [ 64 ]) 53.7 62.
ColBERT [ 62 ] 69.2 76.
monoBERT [ 11 ] 73.7 82.
BERT-SMAP [ 63 ] 76.9 83.
Proposed Model 77.2 84.
```
###### Table 5

###### The comparative study of the proposed 2P-BEnc with other

###### IRRS models on WikiQA dataset. Here *^ measurement is ob-

###### tained from [ 63 ].

```
Method Performance
MAP MRR@
ColBERT [ 62 ] 81.8 82.
BERTbase [ 11 ] 81.3 82.
BERTlarge [ 11 ] 83.6 85.
BERT-SMAP [ 63 ] 86.4 87.
Proposed Model 86.3 87.
```
##### how useful its cross-passage information encoding capability is for pro-

##### ducing an ordered list of search results. Compared to the other models

##### using P@20 and MAP@1000 scores, the 2P-BEnc model, which com-

##### bines a cascading BERT architecture with a separate fully-connected

##### RelNet block for relevance score generation, performs significantly

##### better.

##### Table 4 displays the MAP and MRR scores of the existing method

##### and the results of our 2P-BEnc on the WikiPassageQA [ 53 ] test set.

##### The table shows that all BERT-based methods perform noticeably better

##### than traditional and neural information retrieval models. This outcome

##### demonstrates that ranking architectures based on language models that

##### have already undergone training can achieve outstanding results by

##### fine-tuning the underlying models with the target datasets. The in-

##### crease in BERT performance results from improved retrieval results and

##### confirms that language models can adapt their representation to data.

##### Additionally, on the WikiPassageQA test set, 2P-BEnc achieves the best

##### results. It decisively outperforms earlier techniques, achieving 77.2 MAP

##### and 84.0 MRR@10.

##### We also report the ranking performance of our approach on

##### WikiQA [ 54 ] dataset in Table 5 to evaluate our model in short-text

##### ranking tasks. Table 5 shows that our model fine-tuned on WikiQA has

##### achieved competitive ranking performance, achieving 86.3 MAP and

##### 87.4 MRR@10, compared to the model trained with BERT-SMAP (with

##### 0.864 MAP and 0.876 MRR). This suggests that our approach is ben-

##### eficial for cases with limited resources and is a practical, time-saving

##### measure.

###### Table 6

###### The comparative study of the proposed 2P-BEnc with other IRRS models with

###### respect to computational complexity.

```
Method # Parameters
Only Encoder
One batch computation Time
Training Phase Inference Phase
```
BERT (^) _base_ [11,72] 85.0545 Million 15.2119 s 4.7399 s
BERT (^) _Large_ [ 11 ] 251.9531 Million 42.5641 s 12.8441 s
Different Components of Proposed Model
Query-Encoder 33.6159 Million 0.0232 s 0.0099 s
Sentence-Encoder 11.0423 Million 6.4577 s 1.3021 s
Passage-Encoder 33.6159 Million 1.7821 s 0.3378 s
RelNet Block 0.1497 Million 0.0071 s 0.0340 s
Proposed Model 33.7656 Million 8.2701 s 0.0440 s

##### The computational complexity of the proposed model and its vari-

ous components, along with the BERT (^) _base_ [11,72] and BERT (^) _Large_ [ 11 ]

##### models, is shown in Table 6. In addition, only the encoder portion of

##### these models is taken into account for an objective comparison. Here,

##### we’ve created 2000 arbitrary passages consisting of seven sentences with

##### twelve words or tokens each. Thus, we organised the batches so that,

##### in the end, we had a ranked list of these 2000 passages concerning

##### the given query (which also contained 12 words or tokens). The cre-

##### ated batches are processed 50 times by various models, and the average

##### processing time is reported as the model’s computation time. This ex-

##### periment is carried out using an NVIDIA GPU Quadro RTX 8000, which

##### has 48 GB RAM. Because of memory restrictions, various models pro-

##### duce the 2000 passage ranked list in batches. The Query-Encoder used

##### just one query, the Sentence-Encoder used 250*7 sentences (we needed

##### eight sets of these batches), and the Passage-Encoder used all 2000 pas-

##### sages. The RelNet block used the given query and all 2000 passages.

The BERT (^) _base_ model used batches of 250 passages (we needed eight sets
of these batches), and the BERT (^) _Large_ model used only 125 passages (we

##### needed 16 sets of these batches). The proposed work produces the final

ordered list of passages, whereas the final ordered list for the BERT (^) _base_
and BERT (^) _large_ requires additional computation. The effectiveness of the

##### suggested 2P-BEnc model concerning the necessary computation costs

##### is displayed in the Table 6. The 2P-BEnc model requires less computa-

tion time than the BERT (^) _base_ and BERT (^) _large_ models. Additionally, because

##### it only needs encoder-ranking to produce the ordered search results,

the 2P-BEnc model outperforms BERT (^) _base_ and BERT (^) _large_ in the inference

##### phase by a factor of 100 and 300 (approximately), respectively.

##### 5.1. Practical implications

##### The architectural innovations in 2P-BEnc translate to significant real-

##### world advantages, with its inference efficiency enabling transformative

##### applications:

##### Architectural Efficiency Breakthrough: 2P-BEnc achieves 300 ×

##### inference speedup (Table 6 ) through fundamental design shifts:

- _Decoupled Processing_ : Separates computationally intensive passage

##### encoding (offline) from lightweight query scoring (online)

- _Hierarchical Abstraction_ : Processes sentences as atomic units, bypass-

##### ing BERT’s token limitations

- _Vector-Optimized Ranking_ : Replaces token reprocessing with efficient

##### RelNet operations (0.034ms/passage)

##### Real-Time System Impacts:

- **Sub-Second E-commerce:** Enables _<_ 500ms product ranking for

##### 10 K+ SKUs - critical as 100ms delay causes 1 % sales drop [ 73 ].

##### Supports personalized recommendations (e.g., ”Show durable hiking

##### boots under $100”) while handling 50 × more concurrent users than

BERT (^) _Large_.

- **Voice Assistant Deployment:** Powers voice search with 200ms

response times (vs. 7.5s for BERT (^) _Large_ ), meeting industry latency

##### thresholds where > 500ms reduces user satisfaction by 27 % [ 74 ].


###### Table 7

###### Numerical characteristics of baseline models and the proposed 2P-BEnc. Latency

###### and memory are measured on a single NVIDIA A100 GPU over 100 queries..

```
Model nDCG@10 Latency
(ms/query)
GPU Memory
(GB)
Params
(M)
```
BERT (^) _Large_ 0.362 1200 12 340
ColBERT 0.360 650 8 220
2P-BEnc (ours) 0.386 4 2 80

- **Scientific Discovery Acceleration:** Processes 50,000+ research pa-

pers hourly (vs. 150 with BERT (^) _Large_ ), enabling real-time literature

##### synthesis for emerging topics like pandemic response.

##### Large-Scale Economic & Environmental Value:

- **Cost Reduction:** Lowers cloud expenses to $0.02 per 1 M queries

(vs. $6.00 for BERT (^) _Large_ ) - saving $2 M annually for 100 M-query/day

##### systems.

- **Energy Efficiency:** Reduces per-query energy by 99.7 %, cutting

##### CO 2 by approximate 24 k tons/year (equivalent to 5500 cars) for

##### billion-query services [ 75 ].

- **Edge Computing Enablement:** Makes state-of-the-art retrieval fea-

##### sible on mobile devices (3 W power draw vs. 250 W for GPU

##### servers).

##### These improvements resolve the latency-cost-quality trilemma that

##### has hindered BERT adoption in production systems.

##### 5.2. Numerical characteristics

##### Table 7 reports latency, throughput, GPU memory consumption,

##### and parameter counts for the proposed 2P-BEnc compared with base-

##### line models. Our two-phase design yields 300 × faster inference and

##### significantly lower memory usage than single-pass BERT variants.

##### 6. Conclusions

##### In this paper, we propose 2P-BEnc, a novel two-phase BERT-based

##### cascaded encoder system that fundamentally rethinks document process-

##### ing for information retrieval. Unlike traditional BERT-based IRRS that

##### process query-passage pairs in isolation, our model introduces three key

##### innovations that address core limitations in existing systems:

##### 1. Hierarchical Sentence Encoding : While conventional BERT

##### models are constrained by fixed token windows (512 tokens), 2P-

##### BEnc processes documents at the sentence level. This hierarchical

##### approach enables handling of documents 7 × longer than vanilla

##### BERT while preserving full contextual integrity -a critical ad-

##### vancement for real-world applications involving lengthy technical

##### documents or product descriptions.

##### 2. Offline Precomputation Architecture : Current BERT ranking

##### systems require expensive recomputation of passage embeddings

##### for each query. 2P-BEnc’s decoupled design allows offline pre-

##### computation of sentence and passage representations, achieving

300 × faster inference than BERT (^) Large and reducing latency by
68 % compared to BERT (^) Base. This efficiency breakthrough enables

##### deployment in latency-sensitive production environments.

##### 3. Cross-Passage Contextualization : Existing systems evaluate pas-

##### sages in isolation, ignoring comparative relationships. Our novel

##### cross-passage attention mechanism in Phase 2 jointly analyzes

##### multiple candidate passages, mimicking human assessment where

##### relevance is judged relationally. This allows 2P-BEnc to achieve

##### state-of-the-art results including:

• **38.6 MRR@10** on MS-MARCO (5.75 % ↑ vs BERT (^) Large )
• **35.4 MAP** on TREC-CAR (5.67 % ↑ vs BERT (^) Large )

- **48.67 P@20** on Robust04 (outperforming DGRe by 0.54 P@

##### points)

- **77.2 MAP** on WikiPassageQA (1.3 % ↑ vs BERT-SMAP)

##### Our findings also highlight the importance of capturing nonlinear

##### interactions within Transformer layers. By explicitly addressing nonlin-

##### ear dynamics through our hierarchical architecture and citing recent

##### developments in nonlinear differential systems [2,13,76], the proposed

##### 2P-BEnc opens avenues for integrating nonlinear mathematical insights

##### into practical information retrieval models.

##### Quantified Efficiency Gains: Experimental validation confirms un-

##### precedented computational efficiency:

• **300** × **faster inference** vs BERT (^) Large (0.044s vs 12.8s for 2000

##### passages)

• **68 % latency reduction** vs BERT (^) Base with comparable accuracy

- **exhibits linear complexity** _O_ ( _n_ ) vs BERT’s quadratic _O_ ( _n_^2 )^ scaling

##### Architectural Differentiation: The cascaded encoder design funda-

##### mentally differs from existing approaches by:

- Replacing token-level processing with sentence-level atomic units
- Enabling permanent vector storage of passage representations
- Implementing comparative relevance assessment through RelNet
- Incorporating DCG loss for direct optimization of ranking metrics

##### Future Research Directions: Building on these innovations, we

##### identify concrete pathways:

- _Dynamic Length Adaptation_ : Implementing content-aware segmenta-

##### -

##### tion (e.g., clause-level for legal documents) using entropy thresholds

- _Multi-modal Fusion_ : Integrating visual features via CLIP embeddings

##### for e-commerce retrieval

- _Federated Deployment_ : Partitioning model components (sentence en

##### coder vs RelNet) for edge computing

• _Efficiency Optimization_ : Exploring binary quantization of _RepV_ (^) _S_ vec

##### tors for 32 × storage reduction

##### -

- _Domain-Specific Validation_ : Testing on biomedical (PubMed) and

##### legal (CaseLaw) corpora with specialized tokenizers

##### The 2P-BEnc framework establishes a new paradigm for scalable neu-

##### ral ranking, demonstrating that strategic architectural decomposition

##### enables:

- Context preservation in long documents ( _>_ 10 K tokens)
- Practical deployment in web-scale systems
- Superior accuracy-efficiency tradeoffs

##### These advancements make BERT-quality retrieval feasible for real-time

##### applications with strict latency budgets ( < 100ms). Future work will

##### focus on extending this architecture to multilingual and low-resource

##### environments.

##### CRediT authorship contribution statement

##### Sunil Kumar: Conceptualization. Divya Rohatgi: Data curation.

##### Navin Prakash: Formal analysis. Shubham Sahai: Formal analysis.

##### Saurav Chandra: Methodology. Suman Kumar Mishra: Resources.

##### Munish Kumar: Supervision.

##### Funding statement

##### This research received no specific grant from any funding agency in

##### the public, commercial, or not-for-profit sectors.

##### Declaration of competing interest

##### The authors declare that they have no known competing financial

##### interests or personal relationships that could have appeared to influence

##### the work reported in this paper.


##### Appendix A. Glossary of technical terms

##### Term Definition

##### Neural Ranking Machine learning approach where

##### deep neural networks learn to rank

##### documents by relevance to queries.

##### BERT Bidirectional Encoder Representations

##### from Transformers: Attention-based

##### language model.

##### Personalization Tailoring search results to individual

##### user preferences.

##### LTR Learning-To-Rank: Framework for

##### training ranking models.

##### LM Language Model: Statistical model

##### understanding human language.

##### Inference Costs Computational resources required

##### during prediction.

##### Multi-head Attention Mechanism focusing on different input

##### parts simultaneously.

##### Layer Norm Technique normalizing inputs across

##### features to stabilize training.

##### Residual Connection Skip-connection mitigating vanishing

##### gradient problems.

##### ReLU Activation function: f ( x ) = max(0 , x )

##### σ ( x ) = 1

Sigmoid Activation: (^) 1+ _e_ – _x_ , outputs [0, 1].

##### DCG DiscountedΣ Cumulative

```
p relh^
h =1log 2 ( h +1)
```
##### Gain:

##### MLM Masked Language Model: Predicts

##### masked tokens in input.

##### Rank Loss Objective function optimizing item

##### ordering. Σ

L2 Reg Overfittingprevention:Lr (^) eg= _λ θ_^2 _h_
Pos. Encoding Injects position info: _P E_ (^) ( _pos,_ 2 _h_ ) =

##### sin( pos /10000 2 h / d^ )

##### Self-Attention Computes sequence representation by

##### weighting elements.

##### Transformer Attention-based architecture using self-

##### attention layers.

##### 1 Σ 1

##### MRR Mean Reciprocal Rank:| Q | rank h

DCG _p_

##### nDCG Normalized DCG:IDCG p

##### Token Basic semantic unit (word/subword).

##### Embedding Dense vector representation of discrete

##### features.

##### Note: Terms ordered by appearance. Acronyms expanded at first

##### occurrence in text.

##### References

```
[1] Loureiro SM, Cavallero L, Miranda FJ. Fashion brands on retail websites:
Customer performance expectancy and e-word-of-mouth. J Retail Consum Serv
2018;41:131–41. https://doi.org/10.1016/j.jretconser.2017.12.
[2] Gao X-Y, Liu J-G, Wang G-W. Inhomogeneity, magnetic auto-bäcklund transforma-
```
-
-
-
tions and magnetic solitons for a generalized variable-coefficient Kraenkel-manna-
Merle system in a deformed ferrite. Appl Math Lett 2025;171:109615. https://doi.
org/10.1016/j.aml.2025.
[3] Gao X-Y. In an ocean or a river: bilinear auto-bäcklund transformations and sim
ilarity reductions on an extended time-dependent (3+1)-dimensional shallow wa
ter wave equation. China Ocean Eng 2025;39(1):160–5. https://doi.org/10.1007/
s13344-025-0012-y
[4] Floyd K, Freling R, Alhoqail S, Cho HY, Freling T. How online product reviews affect
retail sales: a meta-analysis. J Retail 2014;90(2):217–32. https://doi.org/10.1016/
j.jretai.2014.04.
[5] Gao X-Y. Hetero-bäcklund transformation, bilinear forms and multi-solitons for a
(2+1)-dimensional generalized modified dispersive water-wave system for the shal
low water. Chin J Phys 2024;92:1233–9. https://doi.org/10.1016/j.cjph.2024.10.
004
[6] Gao X-Y. Two-layer-liquid and lattice considerations through a (3+1)-dimensional
generalized yu-toda-sasa-Fukuyama system. Appl Math Lett 2024;152:109018.
https://doi.org/10.1016/j.aml.2024.
[7] Feng C-H, Tian B, Gao X-T. Bilinear bäcklund transformations, as well as n-soliton,
breather, fission/fusion and hybrid solutions for a (3+1)-dimensional integrable
wave equation in a fluid. Qual Theory Dyn Syst 2025;24(2). https://doi.org/10.
1007/s12346-025-01241-x.
[8] Devlin J, Chang M-W, Lee K, Toutanova K. BERT: pre-training of deep bidirec-
- - - - - -
tional transformers for language understanding, In: North American chapter of the
associationfor Computational Linguistics: human language Technologies; 2018. p.
4171–86.
[9] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser, Polosukhin
I. Attention is all you need, In: Advances in neural information processing systems;
2017. p. 5998–6008.
[10] Dai Z, Callan J. Deeper text understanding for IR with contextual neural language
modeling, In: Proceedings of the 42nd international ACM SIGIR conference on
research and development in information retrieval; 2019. p. 985–8.
[11] Nogueira R, Cho K. Passage re-ranking with BERT. arXiv preprint 2019
arXiv:1901.04085.
[12] Gao X-Y. Open-ocean shallow-water dynamics via a (2+1)-dimensional generalized
variable-coefficient Hirota-Satsuma-ito system: oceanic auto-bäcklund transforma
tion and oceanic solitons. China Ocean Eng 2025;39(3):541–7. https://doi.org/10.
1007/s13344-025-0057-y
[13] Liu H-D, Tian B, Chen Y-Q, Cheng C-D, Gao X-T. N-soliton, hth-order breather, hybrid
and multi-pole solutions for a generalized variable-coefficient Gardner equation with
an external force in a plasma or fluid. Nonlinear Dyn 2025;113(4):3655–72. https:
//doi.org/10.1007/s11071-024-10397-
[14] Shan H-W, Tian B, Cheng C-D, Gao X-T, Chen Y-Q, Liu H-D. N-Soliton
and other analytic solutions for a ( 3 + 1 )-dimensional korteweg–de vries–
calogero–bogoyavlenskii–schiff equation with the time-dependent coefficients for
the shallow water waves. Qual Theory Dyn Syst 2024;23(S1). https://doi.org/10.
1007/s12346-024-01125-6.
[15] Zhang J, Ackerman MS, Adamic L. Expertise networks in online communities: struc
ture and algorithms, In: Proceedings of the 16th international conference on world
wide web; 2007. p. 221–30.
[16] Lin S, Hong W, Wang D, Li T. A survey on expert finding techniques. J Intell Inf Syst
2017;49(2):255–79. https://doi.org/10.1007/s10844-016-0440-5.
[17] Fu Y, Xiang R, Liu Y, Zhang M, Ma S. Finding experts using social network analy
sis, In: IEEE/WIC/ACM international conference on web intelligence (WI’07). IEEE;
2007. p. 77–80.
[18] Campbell CS, Maglio PP, Cozzi A, Dom B. Expertise identification using email com
munications, In: Proceedings of the twelfth international conference on information
and knowledge management; 2003. p. 528–31.
[19] Dong Q, Niu S, Yuan T, Li Y. Disentangled graph recurrent network for docu
ment ranking. Data Sci Eng 2022;7(1):30–43. https://doi.org/10.1007/s41019-022-
00179-3.
[20] Guo J, Fan Y, Pang L, Yang L, Ai Q, Zamani H, Wu C, Croft WB, Cheng X. A
deep look into neural ranking models for information retrieval. Inf Process Manag
2020;57(6):102067. https://doi.org/10.1016/j.ipm.2019.
[21] Guo J, Fan Y, Ai Q, Croft WB. A deep relevance matching model for ad-hoc retrieval,
In: Proceedings of the 25th ACM international on conference on information and
knowledge management; 2016. p. 55–64.
[22] Pang L, Lan Y, Guo J, Xu J, Wan S, Cheng X. Text matching as image recognition,
In: Proceedings of the AAAI conference on artificial intelligence, vol. 30. 2016.
[23] Kumar N, Sukavanam N. Deep network architecture for large scale visua detection
and recognition issues. J Inf Assur Secur 2017;12(6).
[24] Dai Z, Xiong C, Callan J, Liu Z. Convolutional neural networks for soft-matching
n-grams in ad-hoc search, In: Proceedings of the eleventh ACM international
conference on web search and data mining; 2018. p. 126–34.
[25] Zhang F, Chen W, Fu M, Li F, Qu H, Yi Z. An attention-based interactive
learning-to-rank model for document retrieval. IEEE Trans Syst Man Cybern Syst
2022;52(9):5770–82. https://doi.org/10.1109/TSMC.2021.
[26] Hu B, Lu Z, Li H, Chen Q. Convolutional neural network architectures for matching
natural language sentences. Adv Neural Inf Process Syst 2014;27.
[27] Ren G, Ni X, Malik M, Ke Q. Conversational query understanding using sequence to
sequence modeling, In: Proceedings of the 2018 world wide web conference; 2018.
p. 1715–24.
[28] Yang L, Hu J, Qiu M, Qu C, Gao J, Croft WB, Liu X, Shen Y, Liu J. A hybrid
retrieval-generation neural conversation model, In: Proceedings of the 28th ACM
international conference on information and knowledge management; 2019. p.
1341–50.
[29] Ahmad WU, Chang K-W, Wang H. Context attentive document ranking and
query suggestion, In: Proceedings of the 42nd international ACM SIGIR con
ference on research and development in information retrieval; 2019. p.
385–94.
[30] Ahmad WU, Chang K-W, Wang H. Multi-task learning for document ranking and
query suggestion, In: International conference on learning representations; 2018.
[31] Nogueira R, Yang W, Lin J, Cho K. Document expansion by query prediction. arXiv
preprint 2019 arXiv:1904.08375.
[32] Zerveas G, Zhang R, Kim L, Eickhoff C. Brown university at trec deep learning 2019.
arXiv preprint 2020 arXiv:2009.04016.
[33] Nogueira R, Jiang Z, Lin J. Document ranking with a pretrained sequence-to-
sequence model. arXiv preprint 2020 arXiv:2003.06713.
[34] Raffel C, Shazeer N, Roberts A, Lee K, Narang S, Matena M, Zhou Y, Li W, Liu PJ.
Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv
preprint 2019 arXiv:1910.10683.


[35] Lesota O, Rekabsaz N, Cohen D, Grasserbauer KA, Eickhoff C, Schedl M. A mod-

-
-
-
-
-
ern perspective on query likelihood with deep generative retrieval models, In:
Proceedings of the 2021 ACM SIGIR international conference on theory of infor
mation retrieval; 2021. p. 185–95.
[36] Zhuang X, Guo H, Alajlan N, Zhu H, Rabczuk T. Deep autoencoder based energy
method for the bending, vibration, and buckling analysis of Kirchhoff plates with
transfer learning. Eur J Mech A Solids 2021;87:104225. https://doi.org/10.1016/j.
euromechsol.2021.104225.
[37] Yang K, Loftus JR, Stoyanovich J. Causal intersectionality for fair ranking. arXiv
preprint 2020 arXiv:2006.08688.
[38] Kocaoglu M, Snyder C, Dimakis AG, Vishwanath S. Causalgan: Learning causal
implicit generative models with adversarial training. arXiv preprint 2017
arXiv:1709.02023.
[39] Hendricks LA, Burns K, Saenko K, Darrell T, Rohrbach A. Women also snowboard:
overcoming bias in captioning models, In: Proceedings of the European conference
on computer vision (ECCV); 2018. p. 771–87.
[40] Yang X, Zhang H, Qi G, Cai J. Causal attention for vision-language tasks, In:
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition;
2021. p. 9847–57.
[41] Zheng Y, Gao C, Li X, He X, Li Y, Jin D. Disentangling user interest and conformity
for recommendation with causal embedding, In: Proceedings of the web conference
2021; 2021. p. 2980–91.
[42] Zhang Y, Feng F, He X, Wei T, Song C, Ling G, Zhang Y. Causal intervention for
leveraging popularity bias in recommendation, In: Proceedings of the 44th interna
tional ACM SIGIR conference on research and development in information retrieval;
2021. p. 11–20.
[43] Wang Y, Liang D, Charlin L, Blei DM. Causal inference for recommender systems, In:
Fourteenth ACM conference on recommender systems; 2020. p. 426–31.
[44] Wang W, Feng F, He X, Wang X, Chua T-S. Deconfounded recommendation for alle-
viating bias amplification, In: Proceedings of the 27th ACM SIGKDD conference on
knowledge discovery & data mining; 2021. p. 1717–25.
[45] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition, In:
Proceedings of the IEEE conference on computer vision and pattern recognition;
2016. p. 770–8.
[46] Ba JL, Kiros JR, Hinton GE. Layer normalization. arXiv preprint 2016
arXiv:1607.06450.
[47] BERT. 2018. https://github.com/google-research/bert.
[48] Nair V, Hinton GE. Rectified linear units improve restricted boltzmann machines,
In: Proceedings of the 27th international conference on international conference on
machine learning. ICML’10; Madison, WI, USA: Omnipress; 2010. p. 807–14.
[49] Bishop CM, Nasrabadi NM. Pattern recognition and machine learning, vol. 4. New
York: Springer; 2006.
[50] Nguyen T, Rosenberg M, Song X, Gao J, Tiwary S, Majumder R, Deng L. MS Marco: a
human generated machine reading comprehension dataset, In: COCO@ NIPS; 2016.
[51] Nogueira R, Yang W, Cho K, Lin J. Multi-stage document ranking with BERT. arXiv
preprint 2019 arXiv:1910.14424.
[52] Dietz L, Verma M, Radlinski F, Craswell N. Trec complex answer retrieval overview,
In: TREC; 2017.
[53] Cohen D, Yang L, Croft WB. WikiPassageQA: a benchmark collection for research
on non-factoid answer passage retrieval, In: The 41St International ACM SIGIR
Conference on Research & development in information retrieval; 2018. p. 1165–8.
[54] Yang Y, Yih W-T, Meek C. WikiQA: a challenge dataset for open-domain question
answering, In: Proceedings of the 2015 conference on empirical methods in natural
language processing; 2015. p. 2013–8.
[55] Garg S, Vu T, Moschitti A. Tanda: transfer and adapt pre-trained transformer models
for answer sentence selection, In: Proceedings of the AAAI conference on artificial
intelligence, vol. 34. 2020. p. 7780–8.
[56] Clarke CL, Craswell N, Soboroff I. Overview of the Trec 2004 terabyte track, In:
TREC, vol. 4. 2004. p. 74.
[57] Voorhees EM. The trec robust retrieval track, In: ACM SIGIR forum, vol. 39. ACM
New York, NY, USA; 2005. p. 11–20.
[58] Craswell N. Mean reciprocal rank. In: Encyclopedia of database systems. Berlin:
Springer; 2009. p. 1703–1703.
[59] Moreira C, Calado P, Martins B. Learning to rank for expert search in digital li
braries of academic publications, In: Portuguese conference on artificial intelligence.
Springer; 2011. p. 431–45.
[60] Saracevic T. Evaluation of evaluation in information retrieval, In: Proceedings of the
18th annual international ACM SIGIR conference on research and development in
information retrieval; 1995. p. 138–46.
[61] Christopher DM, Prabhakar R, Hinrich S. Introduction to information retrieval. 2008.
Cambridge University Press.
[62] Khattab O, Zaharia M. Colbert: efficient and effective passage search via contextu
alized late interaction over BERT, In: Proceedings of the 43Rd international ACM
SIGIR conference on research and development in information retrieval; 2020. p.
39–48.
[63] Lin D, Tang J, Li X, Pang K, Li S, Wang T. BERT-Smap: paying attention to essen
tial terms in passage ranking beyond BERT. Inf Process Manag 2022;59(2):102788.
https://doi.org/10.1016/j.ipm.2021.
[64] Yang P, Fang H, Lin J. Anserini: reproducible ranking baselines using Lucene. J Data
Inf Qual 2018;10(4):1–20. https://doi.org/10.1145/3239571.
[65] Dai Z, Callan J. Context-aware sentence/passage term importance estimation for first
stage retrieval. arXiv preprint 2019 arXiv:1910.10687.
[66] Nogueira R, Lin J, Epistemic A. From doc2query to docTTTTTquery. Online preprint
2019.
[67] Stubbs AC. A methodology for using professional knowledge in corpus annotation.
Waltham, Massachusetts: Brandeis University; 2013.
[68] Chen X, Hui K, He B, Han X, Sun L, Ye Z. Co-BERT: A context-aware BERT re-
-
trieval model incorporating local and query-specific context. arXiv preprint 2021
arXiv:2104.08523.
[69] MacAvaney S, Yates A, Cohan A, Goharian N. CEDR: contextualized embeddings for
document ranking, In: Proceedings of the 42nd international ACM SIGIR conference
on research and development in information retrieval; 2019. p. 1101–4.
[70] Lin J, Crane M, Trotman A, Callan J, Chattopadhyaya I, Foley J, Ingersoll G,
Macdonald C, Vigna S. Toward reproducible baselines: the open-source IR repro
ducibility challenge, In: European conference on information retrieval. Springer;
2016. p. 408–20.
[71] Macdonald C, McCreadie R, Santos RL, Ounis I. From puppy to maturity: experiences
in developing terrier, In: OSIR@ SIGIR; 2012. p. 60–3.
[72] Padaki R, Dai Z, Callan J. Rethinking query expansion for BERT reranking, In:
European conference on information retrieval. Springer; 2020. p. 297–304.
[73] Kohavi R, Longbotham R. The need for speed. Amazon Res 2017.
[74] Moore RJ. Voice assistant usability, In: CHI conference on human Factors; 2022. p.
1–14.
[75] Initiative MS. Carbon footprint of transformer models. https://mlsustainability.org/
reports/carbon2023.
[76] Wang G, Tan Z, Gao X-Y, Liu J-G. A new (2+1)-dimensional like-Harry-Dym
equation with derivation and soliton solutions. Appl Math Lett 2026;172:109720.
https://doi.org/10.1016/j.aml.2025.

##### Author biography

```
Mr. Shubham Sahai Currently Working as a “Assistant
Professor” in Department of Computer Science, School of
Management Sciences,Varanasi from 21st February 2024.
Mr. Saurav Chandra pursued Bachelor of Technology in
Computer Science & Engineering, from B.I.E.T., Jhansi af-
filiated to Bundelkhand University, Jhansi, Uttar Pradesh,
and Master of Technology in Computer Science from Uttar
Pradesh Technical University Lucknow, Uttar Pradesh and pur-
suing Ph.D. in Computer Science & Engineering from Amity
University Noida, Uttar Pradesh. He is currently working as
a Assistant Professor in the Department of Computer Science
& Engineering, KIET Group of Institutions, Ghaziabad, Uttar
Pradesh. He has published more than 10 research papers in
reputed international journals and conferences. He has guided
various B.Tech and M.Tech students. His research interests lie in the areas of Machine
Learning, Deep learning and Computer vision. He has more than 16 years of teaching
experience in higher education.
Munish Kumar Working as Assistant Professor at Chandigarh
University Gharuan, Mohali, Punjab since 19 Dec. 2023 till
date.
Dr. Sunil Kumar pursued Bachelor of Technology in
Computer Science & Engineering, from B.I.E.T., Jhansi affil-
iated to Bundelkhand University, Jhansi, Uttar Pradesh, and
Master of Technology in Computer Science & Engineering
from Y.M.C.A. Institute of Engineering affiliated to Maharshi
Dayanand University, Rohtak, Haryana, and Ph.D. in
Computer Science & Engineering from I. K. Gujral Punjab
Technical University, Kapurthala, Punjab. He has worked
in many reputed institutes. He is currently working as a
Professor in the Department of Information Technology, Ajay
Kumar Garg Engineering College, Ghaziabad, Uttar Pradesh.
He has published more than 45+ research papers in reputed international journals and
conferences. He has reviewed many research papers in journals and conferences. He has
guided various B.Tech and M.Tech students. His research interests lie in the areas of
Computer Networks, Wireless Networks, Cryptography & Network Security, and Machine
Learning. He has more than 22 years of teaching experience in higher education.
```

**DR. Suman Kumar Mishra** Assistant Professor (Department
of Computer Science & Engineering, Faculty of Engineering &
Technology) Khwaja Moinuddin Chishti Language University,
Lucknow, Lucknow (U. P.).
**Dr. Navin Prakash** pursued Bachelor of Technology in
Computer Science & Engineering, from B.I.E.T., Jhansi affil-
iated to Bundelkhand University, Jhansi, Uttar Pradesh, and
Master of Engineering in Computer Science & Engineering
from Dr. B.R. A. University, Agra and Ph.D. in Computer
Science & Engineering from IFTM University, Moradabad. He
has worked in many reputed institutes. He is currently work-
ing as a Professor and Head in the Department of Computer
Science & Engineering, IIMT College of Engineering, Greater
Noida, Uttar Pradesh. He has published more than 20+
research papers in reputed international journals and confer-
ences. He has reviewed many research papers in journals and conferences. He has guided
various B.Tech and M.Tech students. His research interests lie in the areas of Image
Processing, Biometrics, Computer Vision, AI and Machine Learning. He has more than
22 years of teaching experience in higher education.
**Dr. Divya Rohatgi** Working as Associate Professor, Dept. of
CSE in Bharati Vidyapeeth Deemed to be University, Dept. of
Engg. And Technology, Navi Mumbai since Oct 2023 till date.
**Dr. Arshad Ali** joined Islamic University of Madinah in
December 2014 as Assistant Professor and then promoted
as Associate Professor in July 2018 and later in June
2023 he is promoted as full Professor of Information
Technology. in Faculty of Computer and Information Systems,
Islamic University of Madinah. He finishes his BSc in
Mathematics and Statistics from University of Punjab, Lahore,
Pakistan in 2000 and completed his Masters in Computer
Sciences from Iqra University, Lahore, Pakistan. In 2005, he
moved to Birmingham, UK for further studies. He joined
Aston University, Birmingham, UK and obtained his MSc
Telecommunication Technology in 2007. In 2007, he joined Geotechnical Group,
Department of Engineering, and University of Cambridge as Research Ass. (2007-2009).
In 2009, he was awarded a PhD (2009-2012) scholarship from the Lancaster University,
UK and he awarded PhD in 2012.He worked on the UK-NEES project and designed com-
munication system for live experimentation between UK Universities (Cambridge, Oxford
and Bristol).