# NLP-Question-Answering
<ol>
<li> <a href="https://arxiv.org/pdf/1810.04805v2.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding </a></li>

<img width="1141" alt="image" src="https://user-images.githubusercontent.com/56669333/203578100-ea76a748-99b8-4ebe-b40a-ddfd17b26482.png">
   
    
<b><code>Abstract:</code></b> We introduce a new language representa- tion model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language repre- sentation models (Peters et al., 2018a; Rad- ford et al., 2018), BERT is designed to pre- train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a re- sult, the pre-trained BERT model can be fine- tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task- specific architecture modifications.BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art re- sults on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answer- ing Test F1 to 93.2 (1.5 point absolute im- provement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
  
 
<li><a href="https://arxiv.org/pdf/1710.10903v3.pdf"> Graph Attention Networks</a></li>

<img width="934" alt="image" src="https://user-images.githubusercontent.com/56669333/203578668-a61c3f7a-32ef-43b5-8212-6c7d60d3f584.png">
   
   

<b><code>Abstract:</code></b>We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods’ features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix op- eration (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of spectral-based graph neural net- works simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the- art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein- protein interaction dataset (wherein test graphs remain unseen during training).
   
   
<li><a href="https://arxiv.org/pdf/1907.11692v1.pdf"> RoBERTa: A Robustly Optimized BERT Pretraining Approach</a></li>
   
<img width="1327" alt="image" src="https://user-images.githubusercontent.com/56669333/203579812-5ff96aa4-1840-46d6-a68a-cfabe644fce0.png">
   
   
<b><code>Abstract:</code></b>Language model pretraining has led to sig- nificant performance gains but careful com- parison between different approaches is chal- lenging. Training is computationally expen- sive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final re- sults. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparam- eters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the impor- tance of previously overlooked design choices, and raise questions about the source of re- cently reported improvements. We release our models and code
   
   
<li><a href="https://arxiv.org/pdf/1503.08895v5.pdf"> End-To-End Memory Networks</a></li>
   
<img width="1048" alt="image" src="https://user-images.githubusercontent.com/56669333/203580204-8ae75e36-f85e-4eb8-9c44-8e3e60bbff4d.png">
   
   
<b><code>Abstract:</code></b>We introduce a neural network with a recurrent attention model over a possibly large external memory. The architecture is a form of Memory Network [23] but unlike the model in that work, it is trained end-to-end, and hence requires significantly less supervision during training, making it more generally applicable in realistic settings. It can also be seen as an extension of RNNsearch [2] to the case where multiple computational steps (hops) are performed per output symbol. The flexibility of the model allows us to apply it to tasks as diverse as (synthetic) question answering [22] and to language modeling. For the former our approach is competitive with Memory Networks, but with less supervision. For the latter, on the Penn TreeBank and Text8 datasets our approach demonstrates comparable performance to RNNs and LSTMs. In both cases we show that the key concept of multiple computational hops yields improved results.   
   
   
   
   
<li><a href="https://arxiv.org/pdf/1909.11942v6.pdf"> ALBERT: A Lite BERT for Self-supervised Learning of Language Representations</a></li>
   
<img width="1220" alt="image" src="https://user-images.githubusercontent.com/56669333/203581152-e7fdd8bd-0db4-49e0-bda2-0093c4358c1c.png">
   
   
<b><code>Abstract:</code></b> Increasing model size when pretraining natural language representations often re- sults in improved performance on downstream tasks. However, at some point fur- ther model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we present two parameter- reduction techniques to lower memory consumption and increase the training speed of BERT (Devlin et al., 2019). Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer param- eters compared to BERT-large.    
   
   
   
<li><a href="https://arxiv.org/pdf/1802.05365v2.pdf"> Deep contextualized word representations</a></li>
   
<img width="1142" alt="image" src="https://user-images.githubusercontent.com/56669333/203581779-8e963bfe-b704-4e40-bfee-ad8bdb7a7c00.png">
   
   
<b><code>Abstract:</code></b>We introduce a new type of deep contextual- ized word representation that models both (1) complex characteristics of word use (e.g., syn- tax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned func- tions of the internal states of a deep bidirec- tional language model (biLM), which is pre- trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, tex- tual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.  
   
   
<li><a href="https://arxiv.org/pdf/1910.10683v3.pdf"> Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a></li>
   
<img width="1224" alt="image" src="https://user-images.githubusercontent.com/56669333/203582170-eb7fc8c3-0411-4811-9460-5a14709521d4.png">
   
   
   
<b><code>Abstract:</code></b>Transfer learning, where a model is first pre-trained on a data-rich task before being fine- tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code   
   
<li><a href="https://arxiv.org/pdf/1910.13461v1.pdf"> BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</a></li>
   
<img width="1181" alt="image" src="https://user-images.githubusercontent.com/56669333/203582397-096ff90d-f88b-4522-808c-f69342b55828.png">
   
   
   
<b><code>Abstract:</code></b> We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Tranformer-based neural machine translation architecture which, despite its sim- plicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and many other more re- cent pretraining schemes. We evaluate a num- ber of noising approaches, finding the best per- formance by both randomly shuffling the or- der of the original sentences and using a novel in-filling scheme, where spans of text are re- placed with a single mask token. BART is particularly effective when fine tuned for text generation but also works well for compre- hension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state- of-the-art results on a range of abstractive di- alogue, question answering, and summariza- tion tasks, with gains of up to 6 ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine transla- tion, with only target language pretraining. We also report ablation experiments that replicate other pretraining schemes within the BART framework, to better measure which factors most influence end-task performance.  
   
   
<li><a href="https://arxiv.org/pdf/1405.4053v2.pdf"> DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter</a></li>
   
<img width="622" alt="image" src="https://user-images.githubusercontent.com/56669333/203583034-b0fa2d83-25b2-41cc-b2ab-6afee422b83b.png">
   
<b><code>Abstract:</code></b> As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the- edge and/or under constrained computational training or inference budgets remains challenging. In this work, we propose a method to pre-train a smaller general- purpose language representation model, called DistilBERT, which can then be fine- tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device study.
   
   
<li><a href="https://arxiv.org/pdf/1910.01108v4.pdf"> Distributed Representations of Sentences and Documents</a></li>
   
<img width="646" alt="image" src="https://user-images.githubusercontent.com/56669333/203583190-29b11875-9ffc-4ebe-8c8c-db0ae451d4da.png">
   
   
<b><code>Abstract:</code></b> Many machine learning algorithms require the input to be represented as a fixed-length feature vector. When it comes to texts, one of the most common fixed-length features is bag-of-words. Despite their popularity, bag-of-words features have two major weaknesses: they lose the order- ing of the words and they also ignore semantics of the words. For example, “powerful,” “strong” and “Paris” are equally distant. In this paper, we propose Paragraph Vector, an unsupervised algo- rithm that learns fixed-length feature representa- tions from variable-length pieces of texts, such as sentences, paragraphs, and documents. Our algo- rithm represents each document by a dense vec- tor which is trained to predict words in the doc- ument. Its construction gives our algorithm the potential to overcome the weaknesses of bag-of- words models. Empirical results show that Para- graph Vectors outperform bag-of-words models as well as other techniques for text representa- tions. Finally, we achieve new state-of-the-art re- sults on several text classification and sentiment analysis tasks.   
   
<li><a href ="https://arxiv.org/pdf/1611.01603v6.pdf">Bidirectional Attention Flow for Machine Comprehension</a></li> 

 <img width="1004" alt="image" src="https://user-images.githubusercontent.com/56669333/203583739-d5b29f71-da18-4d82-96fb-b09ae3f88403.png">
  
   
<b><code>Abstract:</code></b>Machine comprehension (MC), answering a query about a given context para- graph, requires modeling complex interactions between the context and the query. Recently, attention mechanisms have been successfully extended to MC. Typ- ically these methods use attention to focus on a small portion of the con- text and summarize it with a fixed-size vector, couple attentions temporally, and/or often form a uni-directional attention. In this paper we introduce the Bi-Directional Attention Flow (BIDAF) network, a multi-stage hierarchical pro- cess that represents the context at different levels of granularity and uses bi- directional attention flow mechanism to obtain a query-aware context represen- tation without early summarization. Our experimental evaluations show that our model achieves the state-of-the-art results in Stanford Question Answering Dataset (SQuAD) and CNN/DailyMail cloze test.   
   
   
<li><a href ="https://arxiv.org/pdf/1906.08237v2.pdf">XLNet: Generalized Autoregressive Pretraining for Language Understanding</a></li>
   
<img width="992" alt="image" src="https://user-images.githubusercontent.com/56669333/203583942-ac994032-70dd-4b6a-901b-f7a96e7958fd.png">
   
   
   
<b><code>Abstract:</code></b> With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining ap- proaches based on autoregressive language modeling. However, relying on corrupt- ing the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking   
   
   
   
<li><a href ="https://arxiv.org/pdf/1904.00962v5.pdf">Large Batch Optimization for Deep Learning: Training BERT in 76 minutes</a></li>
   
<img width="1217" alt="image" src="https://user-images.githubusercontent.com/56669333/203584176-cda957ad-c2a7-489d-b56a-3aacd0a036a9.png">
   
   
<b><code>Abstract:</code></b>Training large deep neural networks on massive datasets is computationally very challenging. There has been recent surge in interest in using large batch stochastic optimization methods to tackle this issue. The most prominent algorithm in this line of research is LARS, which by employing layerwise adaptive learning rates trains RESNET on ImageNet in a few minutes. However, LARS performs poorly for attention models like BERT, indicating that its performance gains are not consistent across tasks. In this paper, we first study a principled layerwise adaptation strategy to accelerate training of deep neural networks using large mini-batches. Using this strategy, we develop a new layerwise adaptive large batch optimization technique called LAMB; we then provide convergence analysis of LAMB as well as LARS, showing convergence to a stationary point in general nonconvex settings. Our empirical results demonstrate the superior performance of LAMB across various tasks such as BERT and RESNET-50 training with very little hyperparameter tuning. In particular, for BERT training, our optimizer enables use of very large batch sizes of 32868 without any degradation of performance. By increasing the batch size to the memory limit of a TPUv3 Pod, BERT training time can be reduced from 3 days to just 76 minutes (Table 1). The LAMB implementation is available online  
   
<li><a href ="https://arxiv.org/pdf/1502.05698v10.pdf">Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks</a></li>
   
   
<img width="1081" alt="image" src="https://user-images.githubusercontent.com/56669333/203584456-bb0759a9-2962-416f-b752-539cec294c9e.png">
   
 
<b><code>Abstract:</code></b>One long-term goal of machine learning research is to produce methods that are applicable to reasoning and natural language, in particular building an intelligent dialogue agent. To measure progress towards that goal, we argue for the use- fulness of a set of proxy tasks that evaluate reading comprehension via question answering. Our tasks measure understanding in several ways: whether a system is able to answer questions via chaining facts, simple induction, deduction and many more. The tasks are designed to be prerequisites for any system that aims to be capable of conversing with a human. We believe many existing learning systems can currently not solve them, and hence our aim is to classify these tasks into skill sets, so that researchers can identify (and then rectify) the failings of their systems. We also extend and improve the recently introduced Memory Networks model, and show it is able to solve some, but not all, of the tasks.  
   
   
<li><a href ="https://arxiv.org/pdf/1606.05250v3.pdf">SQuAD: 100,000+ Questions for Machine Comprehension of Text</a></li>
   
<img width="1175" alt="image" src="https://user-images.githubusercontent.com/56669333/203584677-5d0909be-d9ae-492f-b829-e84a807dc7d6.png">
   
   
<b><code>Abstract:</code></b> We present the Stanford Question Answer- ing Dataset (SQuAD), a new reading compre- hension dataset consisting of 100,000+ ques- tions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the cor- responding reading passage. We analyze the dataset to understand the types of reason- ing required to answer the questions, lean- ing heavily on dependency and constituency trees. We build a strong logistic regression model, which achieves an F1 score of 51.0%, a significant improvement over a simple base- line (20%). However, human performance (86.8%) is much higher, indicating that the dataset presents a good challenge problem for future research.   
   
   
<li><a href ="https://arxiv.org/pdf/1706.01427v1.pdf">A simple neural network module for relational reasoning</a></li>
   
<img width="1084" alt="image" src="https://user-images.githubusercontent.com/56669333/203584895-f9d94743-a2b9-45de-8b71-19a3af956b37.png">
   
<b><code>Abstract:</code></b>Relational reasoning is a central component of generally intelligent behavior, but has proven difficult for neural networks to learn. In this paper we describe how to use Relation Networks (RNs) as a simple plug-and-play module to solve problems that fundamentally hinge on relational reasoning. We tested RN-augmented networks on three tasks: visual question answering using a challenging dataset called CLEVR, on which we achieve state-of-the-art, super-human performance; text-based question answering using the bAbI suite of tasks; and complex reasoning about dynamic physical systems. Then, using a curated dataset called Sort-of-CLEVR we show
that powerful convolutional networks do not have a general capacity to solve relational questions, but can gain this capacity when augmented with RNs. Our work shows how a deep learning architecture equipped with an RN module can implicitly discover and learn to reason about entities and their relations.
   
<li><a href ="https://arxiv.org/pdf/1901.08746v4.pdf">BioBERT: a pre-trained biomedical language representation model for biomedical text mining</a></li>
   
<img width="1136" alt="image" src="https://user-images.githubusercontent.com/56669333/203585221-5e679462-88b8-48b3-98a7-eb5a91ad2643.png">
   
   
<b><code>Abstract:</code></b>Motivation: Biomedical text mining is becoming increasingly important as the number of biomedical documents rapidly grows. With the progress in natural language processing (NLP), extracting valuable information from bio- medical literature has gained popularity among researchers, and deep learning has boosted the development of ef- fective biomedical text mining models. However, directly applying the advancements in NLP to biomedical text min- ing often yields unsatisfactory results due to a word distribution shift from general domain corpora to biomedical corpora. In this article, we investigate how the recently introduced pre-trained language model BERT can be adapted for biomedical corpora.   
   
<li><a href ="https://arxiv.org/pdf/2003.10555v1.pdf">ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators</a></li>
   
   
<img width="1012" alt="image" src="https://user-images.githubusercontent.com/56669333/203585534-b887306b-c3a3-48ae-99b5-15304b5f3214.png">
   
   
<b><code>Abstract:</code></b>Masked language modeling (MLM) pre-training methods such as BERT corrupt the input by replacing some tokens with [MASK] and then train a model to re- construct the original tokens. While they produce good results when transferred to downstream NLP tasks, they generally require large amounts of compute to be effective. As an alternative, we propose a more sample-efficient pre-training task called replaced token detection. Instead of masking the input, our approach cor- rupts it by replacing some tokens with plausible alternatives sampled from a small generator network. Then, instead of training a model that predicts the original identities of the corrupted tokens, we train a discriminative model that predicts whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments demonstrate this new pre-training task is more ef- ficient than MLM because the task is defined over all input tokens rather than just the small subset that was masked out. As a result, the contextual representa- tions learned by our approach substantially outperform the ones learned by BERT given the same model size, data, and compute. The gains are particularly strong for small models; for example, we train a model on one GPU for 4 days that outperforms GPT (trained using 30x more compute) on the GLUE natural lan- guage understanding benchmark. Our approach also works well at scale, where it performs comparably to RoBERTa and XLNet while using less than 1/4 of their compute and outperforms them when using the same amount of compute.
   
   
<li><a href ="https://arxiv.org/pdf/2004.04906v3.pdf">Dense Passage Retrieval for Open-Domain Question Answering</a></li>

<img width="1201" alt="image" src="https://user-images.githubusercontent.com/56669333/203585720-b6c9db1c-573d-42a7-a2c7-ce78cef00d2f.png">
   
   
<b><code>Abstract:</code></b>Open-domain question answering relies on ef- ficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented us- ing dense representations alone, where em- beddings are learned from a small number of questions and passages by a simple dual- encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene- BM25 system greatly by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.   
   
<li><a href ="https://arxiv.org/pdf/1804.09541v1.pdf">QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension</a></li>
   
<img width="1063" alt="image" src="https://user-images.githubusercontent.com/56669333/203585947-dbd73175-a15f-42f7-83e9-f9a30ad151e8.png">
   
   
   
<b><code>Abstract:</code></b>Current end-to-end machine reading and question answering (Q&A) models are primarily based on recurrent neural networks (RNNs) with attention. Despite their success, these models are often slow for both training and inference due to the se- quential nature of RNNs. We propose a new Q&A architecture called QANet, which does not require recurrent networks: Its encoder consists exclusively of convolution and self-attention, where convolution models local interactions and self-attention models global interactions. On the SQuAD dataset, our model is 3x to 13x faster in training and 4x to 9x faster in inference, while achieving equiva- lent accuracy to recurrent models. The speed-up gain allows us to train the model with much more data. We hence combine our model with data generated by back- translation from a neural machine translation model. On the SQuAD dataset, our single model, trained with augmented data, achieves 84.6 F1 score1 on the test set, which is significantly better than the best published F1 score of 81.8.   
   
   
   
<li><a href ="https://arxiv.org/pdf/1904.01201v2.pdf">Habitat: A Platform for Embodied AI Research</a></li>
   
   
<img width="1146" alt="image" src="https://user-images.githubusercontent.com/56669333/203586173-8df32b1a-3514-4c9a-92cd-0e9a963a28a7.png">
   
   
<b><code>Abstract:</code></b>We present Habitat, a platform for research in embodied artificial intelligence (AI). Habitat enables training embod- ied agents (virtual robots) in highly efficient photorealistic 3D simulation. Specifically, Habitat consists of:
(i) Habitat-Sim: a flexible, high-performance 3D sim- ulator with configurable agents, sensors, and generic 3D dataset handling. Habitat-Sim is fast – when rendering a scene from Matterport3D, it achieves several thousand frames per second (fps) running single-threaded, and can reach over 10,000 fps multi-process on a single GPU.(ii) Habitat-API: a modular high-level library for end-to- end development of embodied AI algorithms – defining tasks (e.g. navigation, instruction following, question answering), configuring, training, and benchmarking embodied agents.   
   
   
<li><a href ="https://arxiv.org/pdf/2001.04451v2.pdf">Reformer: The Efficient Transformer</a></li>
   
<img width="971" alt="image" src="https://user-images.githubusercontent.com/56669333/203586411-f3c456b4-304b-4ad0-846c-1b5b12b8b4d3.png">
   
<b><code>Abstract:</code></b>Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transform- ers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L2 ) to O(L log L), where L is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training pro- cess instead of N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.   
   
<li><a href ="https://arxiv.org/pdf/1806.03822v1.pdf">Know What You Don't Know: Unanswerable Questions for SQuAD</a></li>
   
<img width="1379" alt="image" src="https://user-images.githubusercontent.com/56669333/203586687-d89ecfdf-7302-4b70-aa57-bdd1f7cccb2a.png">
   
   
<b><code>Abstract:</code></b>Extractive reading comprehension sys- tems can often locate the correct answer to a question in a context document, but they also tend to make unreliable guesses on questions for which the correct answer is not stated in the context. Existing datasets either focus exclusively on answerable questions, or use automatically generated unanswerable questions that are easy to identify. To address these weaknesses, we present SQuAD 2.0, the latest version of the Stanford Question Answering Dataset (SQuAD). SQuAD 2.0 combines exist- ing SQuAD data with over 50,000 unan- swerable questions written adversarially by crowdworkers to look similar to an- swerable ones.    
   
   
<li><a href ="https://arxiv.org/pdf/1603.01417v1.pdf">Dynamic Memory Networks for Visual and Textual Question Answering</a></li>
   
   
<img width="1342" alt="image" src="https://user-images.githubusercontent.com/56669333/203586881-37556722-8837-44ea-ac8b-b6a01f8ee1a7.png">
   
   
<b><code>Abstract:</code></b>Neural network architectures with memory and attention mechanisms exhibit certain reason- ing capabilities required for question answering. One such architecture, the dynamic memory net- work (DMN), obtained high accuracy on a vari- ety of language tasks. However, it was not shown whether the architecture achieves strong results for question answering when supporting facts are not marked during training or whether it could be applied to other modalities such as images. Based on an analysis of the DMN, we propose several improvements to its memory and input modules. Together with these changes we intro- duce a novel input module for images in order to be able to answer visual questions. Our new DMN+ model improves the state of the art on both the Visual Question Answering dataset and the bAbI-10k text question-answering dataset without supporting fact supervision.
   
   
<li><a href ="https://arxiv.org/pdf/1611.09268v3.pdf">MS MARCO: A Human Generated MAchine Reading COmprehension Dataset</a></li>

<img width="1042" alt="image" src="https://user-images.githubusercontent.com/56669333/203587092-c87c27f2-ddb7-4843-8d9c-b5f2cbb4aef6.png">
   
   
<b><code>Abstract:</code></b>We introduce a large scale MAchine Reading COmprehension dataset, which we name MS MARCO. The dataset comprises of 1,010,916 anonymized questions— sampled from Bing’s search query logs—each with a human generated answer and 182,669 completely human rewritten generated answers. In addition, the dataset contains 8,841,823 passages—extracted from 3,563,535 web documents retrieved by Bing—that provide the information necessary for curating the natural language answers. A question in the MS MARCO dataset may have multiple answers or no answers at all. Using this dataset, we propose three different tasks with varying levels of difficulty: (i) predict if a question is answerable given a set of context passages, and extract and synthesize the answer as a human would (ii) generate a well-formed answer (if possible) based on the context passages that can be understood with the question and passage context, and finally (iii) rank a set of retrieved passages given a question. The size of the dataset and the fact that the questions are derived from real user search queries distinguishes MS MARCO from other well-known publicly available datasets for machine reading comprehension and question-answering. We believe that the scale and the real-world nature of this dataset makes it attractive for benchmarking machine reading comprehension and question-answering models.   


</ol>
