# Cognitive-RAG
A RAG model using graph data for improved question answering through cognition.

you can get raw data from [here](https://zenodo.org/records/7611544/files/MAPLE.zip)
or 
You can directly download the preprocessed graph environment files [here](https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9) and save them to data/processed_data/{data_name}.

The test file data.json is in the data folder.

Run this command to run the baseline rag with any number of hops
```bash
python run_rag.py --dataset maple 
--model claude-3-sonnet-20240229 
--path "/home/ubuntu/cot/data/processed_data/maple/Physics"
--anthropic_key {api_key}
--retrieve_graph_hop {number of hops}
--save_file {path}
``` 
Run this command to run Cognitive-Rag
```bash
python run.py --dataset maple \
--model claude-3-sonnet-20240229 \
--path "/home/ubuntu/cot/data/processed_data/maple/Physics" \
--save_file {path} \
--anthropic_api_key {api_key}
``` 

All the results are stored in the data folder and then run this command to get metric values
```bash
python eval.py 
--result_file {path to result file} 
--model claude-3-sonnet-20240229
--anthropic_key {api_key} 
--output_log {path}
``` 

The metric results of Hop-0,Hop-1,Hop-2 and cognitive-Rag approaches are stored in evaluation_results.json
