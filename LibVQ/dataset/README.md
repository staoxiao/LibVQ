# Dataset

## Data Format
- collection.tsv  
`doc_id \t text_1 \t text_2,... \n`  
For example:
```
0    https://answers.yahoo.com/question/index?qid=20080718121858AAmfk0V      I have trouble swallowing due to MS, can I crush valium & other meds to be easier to swallowll?
1    http://vanrcook.tripod.com/presidentroosevelt.htm       President Roosevelt Led US To Victory In World War 2    "In World War 2, the three great Allied leaders against 
```

- {mode}-queries.tsv  
`query_id, text_1, text_2,... \n`  
For example:
```
0    what does physical medicine do
1    what is a flail chest
```

- {mode}-rels.tsv  
`query_id \t doc_id`  
For example:
```
0    3
0    2022
1    666
```

## Preprocess
```python


```




## DataLoader
```python


```