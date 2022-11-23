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
<li><a href ="https://arxiv.org/pdf/1906.08237v2.pdf">XLNet: Generalized Autoregressive Pretraining for Language Understanding</a></li>
<li><a href ="https://arxiv.org/pdf/1904.00962v5.pdf">Large Batch Optimization for Deep Learning: Training BERT in 76 minutes</a></li>
<li><a href ="https://arxiv.org/pdf/1502.05698v10.pdf">Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks</a></li>
<li><a href ="https://arxiv.org/pdf/1606.05250v3.pdf">SQuAD: 100,000+ Questions for Machine Comprehension of Text</a></li>
<li><a href ="https://arxiv.org/pdf/1706.01427v1.pdf">A simple neural network module for relational reasoning</a></li>
<li><a href ="https://arxiv.org/pdf/1901.08746v4.pdf">BioBERT: a pre-trained biomedical language representation model for biomedical text mining</a></li>
<li><a href ="https://arxiv.org/pdf/2003.10555v1.pdf">ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators</a></li>
<li><a href ="https://arxiv.org/pdf/2004.04906v3.pdf">Dense Passage Retrieval for Open-Domain Question Answering</a></li>
<li><a href ="https://arxiv.org/pdf/1804.09541v1.pdf">QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension</a></li>
<li><a href ="https://arxiv.org/pdf/1904.01201v2.pdf">Habitat: A Platform for Embodied AI Research</a></li>
<li><a href ="https://arxiv.org/pdf/2001.04451v2.pdf">Reformer: The Efficient Transformer</a></li>
<li><a href ="https://arxiv.org/pdf/1806.03822v1.pdf">Know What You Don't Know: Unanswerable Questions for SQuAD</a></li>
<li><a href ="https://arxiv.org/pdf/1603.01417v1.pdf">Dynamic Memory Networks for Visual and Textual Question Answering</a></li>
<li><a href ="https://arxiv.org/pdf/1611.09268v3.pdf">MS MARCO: A Human Generated MAchine Reading COmprehension Dataset</a></li>



</ol>
