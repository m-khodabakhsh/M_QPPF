# M_QPPF
1. Download [MSMARCO collection](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz) ```collection.tsv``` and store it in ```dataset``` repository.
2. We require the query, and their performance (MRR@10). To do so, we create two files named as ```train_query_mrr.tsv``` and ```train_query_mrr.tsv``` for the train and dev sets respectively as follows:
```
    query_text<\t>query_text<\t>query_MRR@10_value
```
   and store them in ```dataset``` repository.

3. Also, we need to prepare two files of ```RetrievedwithRelevancetraink``` and ```RetrievedwithRelevancedevk``` which include the run files for ```top-k``` retrieved documents for queries in MSMARCO train and dev sets and store them in ```dataset``` repository..
      * The runfile should have the folloing format for each query per line:
      ```
         QID<\t>DOCID<\t>relevance grade
      ```
      where relevance grade can be ```1``` for judged relevant documents and ```0``` for others.
 4. run ```train.py```. The trained model will be saved in ```output``` directory.
 5. run ```test.py```. The ranking results will be saved in ```output``` directory in the following format: 
    ```
      QID<\t>Q0<\t>DOCID<\t>rank<\t>predictedScore<\t>M-QPPF
    ```
    To evaluate the results, you can calculate the performans of each query in terms of ```MRR@10```.
