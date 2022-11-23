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
<li><a href="https://arxiv.org/pdf/1802.05365v2.pdf"> Deep contextualized word representation</a>s</li>
<li><a href="https://arxiv.org/pdf/1910.10683v3.pdf"> Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a></li>
<li><a href="https://arxiv.org/pdf/1910.13461v1.pdf"> BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</a></li>
<li><a href="https://arxiv.org/pdf/1405.4053v2.pdf"> DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter</a></li>
<li><a href="https://arxiv.org/pdf/1910.01108v4.pdf"> Distributed Representations of Sentences and Documents</a></li>
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
