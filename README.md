# M_QPPF
1. Download [MSMARCO collection](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz) ```collection.tsv``` and store it in the ```dataset``` folder.
2. We require the queries, and their performance (MRR@10). To do so, we create two files named ```train_query_mrr.tsv``` and ```train_query_mrr.tsv``` for the train and dev set respectively as follows:
```
    query_text<\t>query_text<\t>query_MRR@10_value
```
   and store them in the ```dataset``` folder.

3. Also, we need to prepare two files of ```RetrievedwithRelevancetraink``` and ```RetrievedwithRelevancedevk``` which include the run files for ```top-k``` retrieved documents for queries in MSMARCO train and dev sets and store them in the ```dataset``` folder.
      * The run file should have the following format for each query per line:
      ```
         QID<\t>DOCID<\t>relevance grade
      ```
      where relevance grade can be ```1``` for judged relevant documents and ```0``` for others.
 4. Run ```train.py```. The trained model will be saved in the ```output``` directory.
 5. Run ```test.py```. The ranking results will be saved in the ```output``` directory in the following format: 
    ```
      QID<\t>Q0<\t>DOCID<\t>rank<\t>predictedScore<\t>M-QPPF
    ```
    To evaluate the results, you can calculate the performance of each query in terms of ```MRR@10```.
