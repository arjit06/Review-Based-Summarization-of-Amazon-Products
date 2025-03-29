# Review-Based-Summarization-of-Amazon-Products
Web browsing has emerged beyond mere consumption owing to the amount of time every individual spends on the Internet, a key part of this being online shopping. A major contribution of users in this domain is feedback and suggestions, thereby accumulating a wealth of reviews. These reviews could potentially contain more accurate and descriptive information about the product, since product sellers often tend to write fake reviews and glorify their product to attract clicks. To enhance the user’s decision-making process, these reviews can be thoroughly explored to extract meaningful information. <br>
This study focuses on developing a framework for identifying fake reviews and generating accurate product descriptions, utilizing advanced natural language processing techniques. This study aims to eliminate the impact of fake reviews and enhance user engagement by generating customized product features using product ratings, review texts and sellers’ product descriptions. <br>
This study makes use of an Encoder-only architecture that implements both Self and Cross Attention across the review texts and the product description to separate fake reviews from real ones. An alternative product description is generated based on the identified real reviews and compared against the seller’s product description to evaluate the accuracy between sellers’ claims and user experience.

<br><br>
## Task 1: BUILDING A FAKE REVIEW CLASSIFIER
This task aims at training an Encoder-only Transformer based architecture 1 with a custom Self & Cross-Categorical Attention (SCCA) mechanism. The dataset used for this task contains - Product Descriptions - Product Title & Product Category, Review Descriptions - Review Title and Review Text, the Rating corresponding to each review, and the corresponding label indicating Real (0) or Fake (1) reviews.<br><br>

### Dataset 1: Amazon Reviews Dataset (Real/Fake Label Version):
This dataset provides structured information about product reviews, enabling analysis and modeling tasks related to sentiment analysis, product recommendation, and more. Labels indicating real or fake reviews (Label2 and Label1 respectively) enable the development and evaluation of algorithms for detecting fraudulent or deceptive reviews, which is crucial for maintaining the integrity of online review systems. An example Data sample is given below:- 

![image](https://github.com/arjit06/Review-Based-Summarization-of-Amazon-Products/assets/108218688/9d646dd6-5b6c-4bd4-83b2-725c55d4f4b6) <br><br><br>


### Methodology
The "PRODUCT CATEGORY" and "PRODUCT TITLE" columns are combined using a "[SEP]" token, and "REVIEW TITLE" and "REVIEW TEXT" are combined similarly. Text is processed with contractions expanded, stop words removed, and sequences padded or truncated to specific lengths (CLS and EOS tokens were also used) . FastText embeddings and positional encodings are used to generate representations for an Encoder-only Transformer. The architecture of the model is as follows:- 

![NLP ARCHITECTURE (9)](https://github.com/arjit06/Review-Based-Summarization-of-Amazon-Products/assets/108218688/0dc0d247-5c95-46b0-8fa0-9129c3a455d5)<br><br><br>


The Encoder-Only Transformer consists of multiple encoder layers, each using Self & Cross-Categorical Attention (SCCA). Each encoder layer takes the output from the previous layer, adds cross-attention values and attention masks, and then incorporates a residual input followed by layer normalization. This output is passed through a Feed-Forward Network (FFN), then another residual addition and layer normalization. In the final layer, the [CLS] token output (CLS-Pooling) is extracted for sequence classification, processed through multiple feed-forward layers, and then reduced to a 2D output for logits. Softmax is applied to generate probabilities. The training used Cross Entropy Loss and the Adam optimizer.

![NLP ARCHITECTURE (1)](https://github.com/arjit06/Review-Based-Summarization-of-Amazon-Products/assets/108218688/696d37bf-1170-42f5-94dd-cb2c4eaa9ea0)<br><br><br>

The SCCA (Self & Cross-Categorical Attention) model applies both Self and Cross Attention to capture intra-dependencies within the review text and inter-dependencies between the review text and product descriptions. The Multi-Head SCCA mechanism in each encoder layer computes self-attention on the review text and adds cross-attention, where the queries are from the review text, and the keys and values are from the product description. The cross-attention output is consistent across all encoder layers, with queries, keys, and values generated through linear layers.

![NLP ARCHITECTURE](https://github.com/arjit06/Review-Based-Summarization-of-Amazon-Products/assets/108218688/05e6ed59-625b-449c-bcfa-eb7d7a51047d)<br><br><br>

Self Attention is used to capture patterns in the text itself to detect AI based Fake reviews and cross attention helps to determine irrelevant or inconsistent reviews. <br><br><br>


## Task 2: GENERATING A NEW DESCRIPTION FOR THE PRODUCT BASED ON REAL REVIEWS AND COMPARING THIS WITH THE VENDOR PROVIDED DESCRIPTION
This task focuses on creating product description summaries from authentic reviews and assessing sellers' descriptions by comparing them to the generated summaries using a Semantic Similarity method. The summaries are produced with a fine-tuned BART model specialized for summarizing conversations. Human evaluation of these summaries on random samples indicates that the model effectively condenses reviews into a cohesive summary of the collective opinions. To measure similarity, cosine similarity is calculated between SentenceBERT embeddings of the given product description and the generated summary. <br><br>

### Dataset 2: Amazon Reviews Dataset  (Review Summary Version):
This dataset serves as a valuable resource due to its vast size, extended temporal range, and enriched metadata make it conducive to conducting comprehensive analyses and training robust machine learning models. Additionally, the inclusion of review summaries provides insights into the overall sentiment and key aspects of each review, aiding in tasks such as sentiment analysis and feature extraction. An example Data sample is given below:- 

![image](https://github.com/arjit06/Review-Based-Summarization-of-Amazon-Products/assets/108218688/8a839495-bc61-48aa-b2a8-14e3329a6cea) <br>

The dataset had 30 categories of products and over 22 million reviews in total. For this task, we sampled 5k reviews from each category to obtain a well-represented, smaller version of the dataset. <br><br>


## RESULTS 
The Fake Review Classifier achieves high (75%) confidence in classifying reviews as fake. The classifier is able to accurately identify fake reviews given the context of the product description. The sentence similarity between the generated summaries is evaluated by human evaluation of random samples from the data. It is observed that the products for which reviews are in stark contrast to the product descriptions, show low similarity and vice versa.<br><br>


## CONCLUSION 
The objective of this study is to determine fake reviews and generate quality product descriptions using existing features through opinion mining and advanced model techniques. Our results indicate that current fake reviews appear so realistic that it is challenging for a human to detect them. Fortunately, our models perform very well in this regard, showcasing the effectiveness of our approach.
Additionally, generating correct quality descriptions is crucial in today’s fast-paced world where individuals may not have the time to read through all the reviews to gather information about specific product features.
In future work, we plan to experiment with our model using more diverse datasets and platforms. We can investigate the effectiveness of multi-modal approaches that leverage not only textual data but also other modalities such as images, audio, and video associated with reviews. Furthermore, research efforts could focus on incorporating active learning strategies to enhance the accuracy of our technique and address emerging
challenges in fake review detection and product description generation.


