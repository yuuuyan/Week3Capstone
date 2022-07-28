# Week3Capstone: Semantic Image Search

Organizing COCO data (Joshua)
- create class organizing data
- stores:
    - image ids
    - caption ids
    - caption id --> caption (dict)

Embedding queries and captions (Jayashabari, Jamie)
- process caption/queries by lowercasing text, stripping punctuation, tokenizing (refer to BagOfWords)
- vocab
  - compute IDF for vocab where total caption count = N
- function to embed caption text using GloVe word embeddings

Embedding image descriptors (Kate, Aaron)
- create MyNN model (input: descriptor of img --> output: image embedding)
- extract triple (image ID, caption ID, confuser image ID)
- get margin ranking loss (ourgrad) + accuracy
- train model
  - get caption embedding
  - embed true image + confuser image
  - compute similarities
  - compute loss + accuracy
  - take optimization step
- save/load train model weights (numpy)

Create image matching database (Eric)
- convert img descriptor vector to embedding vector
- create db that matches image id --> image embedding vector
- function that queries db with caption, returns top k matching images
- function that displays k images given URLs
