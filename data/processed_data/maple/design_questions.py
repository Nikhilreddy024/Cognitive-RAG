
import os
import pickle
import random
import json

from collections import defaultdict

domain="Physics" 
data_dir=f"./{domain}"
downstream_dir="~/cot/data/processed_data/maple"

# read processed graph
with open(os.path.join(data_dir,'graph.json'),  'r') as fp:
    graph = json.load(fp)
print(graph.keys())

k = 3
all_generated_data = {} # key: triple (question (str), answer (str)), value: generated data (List)
#%% md
# ### Design questions (one type of question in one cell)
#%% md
# 1-hop question (EASY):
# 1. Who are the authors of paper xxx?
# 2. Where is paper xxx published?
# 
#%%
## question (easy): who are the authors of paper xxx?

random.seed(2023)
print("1-hop")
question = 'Who are the authors of paper "{paper_title}?" '
answer = "{authors}"
generated_data = []

paper_ids = list(graph['paper_nodes'].keys())
random.shuffle(paper_ids)

for paper_id in paper_ids:
    paper_title = graph['paper_nodes'][paper_id]['features']['title']
    author_ids = graph['paper_nodes'][paper_id]['neighbors']['author']
    author_names = [graph['author_nodes'][author_id]['features']['name'] for author_id in author_ids]
    generated_data.append({"paper_title":paper_title, "authors": ', '.join(author_names)})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
random.seed(2024)

question = 'Where is the paper "{paper_title}" published?'
answer = "{venue}"
generated_data = []
print("2-hop")
paper_ids = list(graph['paper_nodes'].keys())
random.shuffle(paper_ids)

for paper_id in paper_ids:
    paper_title = graph['paper_nodes'][paper_id]['features']['title']

    assert len(graph['paper_nodes'][paper_id]['neighbors']['venue']) == 1
    venue_id = graph['paper_nodes'][paper_id]['neighbors']['venue'][0]
    
    venue_name = graph['venue_nodes'][venue_id]['features']['name']
    generated_data.append({"paper_title":paper_title, "venue": venue_name})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%% md
# Multi-hop Reasoning Question (Medium)
#%% md
# 
# 1. Who collaborates with author xxx to write paper xxx?
# 2. who writed both paper xxx and paper xxx?
# 3. How many collaborators does author xxx have in xxx?
# 4. How many papers did xxx and xxx write together?
# 5. Who is the closest collaborator with author xxx?
#%%
## question (medium): Who collaborates with author xxx to write paper xxx?

random.seed(2025)
print("3-hop")
question = 'Who collaborates with author {author_name} to write paper "{paper_title}"?'
answer = "{collaborators}"
generated_data = []

paper_ids = list(graph['paper_nodes'].keys())
random.shuffle(paper_ids)

for paper_id in paper_ids:
    paper_title = graph['paper_nodes'][paper_id]['features']['title']
    author_ids = graph['paper_nodes'][paper_id]['neighbors']['author']
    author_names = [graph['author_nodes'][author_id]['features']['name'] for author_id in author_ids]
    
    if len(author_names) <= 1:
        continue

    random.shuffle(author_names)

    generated_data.append({"author_name": author_names[0],
                       "paper_title": paper_title,
                       "collaborators": ', '.join(author_names[1:])})
    
    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
## question (medium): who writed both paper xxx and paper xxx?

random.seed(2026)

question = 'Who writed both the paper "{paper1_title}" and paper "{paper2_title}"?'
answer = "{authors}"
generated_data = []
print("4-hop")
author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id in author_ids:
    paper_ids = list(graph['author_nodes'][author_id]['neighbors']['paper'])
    random.shuffle(paper_ids)
    if len(paper_ids) < 2:
        continue

    author_list1 = graph['paper_nodes'][paper_ids[0]]['neighbors']['author']
    author_list2 = graph['paper_nodes'][paper_ids[1]]['neighbors']['author']

    if len(set(author_list1) & set(author_list2)) > 1:
        continue

    generated_data.append({"paper1_title": graph['paper_nodes'][paper_ids[0]]['features']['title'],
                            "paper2_title": graph['paper_nodes'][paper_ids[1]]['features']['title'],
                            "authors": graph['author_nodes'][author_id]['features']['name']})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
# Question (medium):Who is the closest collaborator with author xxx?

'''
Closeness is defined in terms of the number of collaboration together. 
The most number of collaboration a pair has, the most closest they are
'''

random.seed(2028)
print("5-hop")
question = "Who is the closest collaborator with author {author_name}? Closeness is defined in terms of the number of collaboration together."
answer = "{collaborator_name}"
generated_data = []

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id in author_ids:
    paper_ids = graph['author_nodes'][author_id]['neighbors']['paper']
    collaborators_by_count = {} #key: collaborator_name, value: paper_counts

    for paper_id in paper_ids:
        collaborator_ids = graph['paper_nodes'][paper_id]['neighbors']['author']
        collaborator_names = [graph['author_nodes'][cid]['features']['name'] for cid in collaborator_ids if cid != author_id]
        
        for collab in collaborator_names:
            if collab not in collaborators_by_count:
                collaborators_by_count[collab] = 0
            collaborators_by_count[collab] += 1

    if len(collaborators_by_count) == 0:
        continue

    sorted_collaborators = sorted(collaborators_by_count.items(), key = lambda item: item[1], reverse = True)
    
    if len(sorted_collaborators) > 1 and sorted_collaborators[0][1] == sorted_collaborators[1][1]:
        continue
    
    author_name = graph['author_nodes'][author_id]['features']['name']
    
    generated_data.append({"author_name": author_name,
                        "collaborator_name": sorted_collaborators[0][0],
                          })

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
# Question (medium): How many collaborators does author xxx have in xxx?

random.seed(2027)
print("6-hop")
question = "How many collaborators does author {author_name} have in {year}"
answer = "{number}"
generated_data = []

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id in author_ids:
    paper_ids = graph['author_nodes'][author_id]['neighbors']['paper']
    collaborators_by_year = defaultdict(set) #key: year, value: author_names

    for paper_id in paper_ids:
        year = graph['paper_nodes'][paper_id]['features']['year']
        collaborator_ids = graph['paper_nodes'][paper_id]['neighbors']['author']
        collaborator_names = [graph['author_nodes'][cid]['features']['name'] for cid in collaborator_ids]
        collaborators_by_year[year].update(collaborator_names)

    author_name = graph['author_nodes'][author_id]['features']['name']
    
    years = [y for y in collaborators_by_year]
    random.shuffle(years)
    
    generated_data.append({"author_name": author_name,
                        "year": years[0],
                        "number": len(collaborators_by_year[years[0]])-1})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
# Question: How many papers did xxx and xxx write together?

random.seed(2028)
print("7-hop")
question = "How many papers did {author_name1} and {author_name2} write together?"
answer = "{number}"
generated_data = []

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id1 in author_ids:
    curr_author_ids = list(graph['author_nodes'].keys())
    random.shuffle(curr_author_ids)
    for author_id2 in curr_author_ids:

        if author_id1 == author_id2: 
            continue
        
        paper_ids1 = graph['author_nodes'][author_id1]['neighbors']['paper']
        paper_ids2 = graph['author_nodes'][author_id2]['neighbors']['paper']

        if len(set(paper_ids1) & set(paper_ids2)) < 2:
            continue

        author_name1 = graph['author_nodes'][author_id1]['features']['name']
        author_name2 = graph['author_nodes'][author_id2]['features']['name']

        generated_data.append({"author_name1": author_name1,
                            "author_name2": author_name2,
                            "number": len(set(paper_ids1) & set(paper_ids2))})
        break
            
    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%% md
# Degree-based reasoning (easy)
# 1. How many papers cite paper xxx?
# 2. How many papers do paper xxx cite?
# 3. How many papers did author xxx write?
#%%
## question (medium): how many paper cite paper xxx?

random.seed(2030)
print("8-hop")
question = 'How many papers cite the paper "{paper_title}"?'
answer = "{num}"
generated_data = []

paper_ids = list(graph['paper_nodes'].keys())
random.shuffle(paper_ids)

for paper_id in paper_ids:
    paper_title = graph['paper_nodes'][paper_id]['features']['title']
    cited_by_id = graph['paper_nodes'][paper_id]['neighbors']['cited_by']
    if len(cited_by_id)  == 0:
        continue
    generated_data.append({"paper_title": paper_title, "num": len(cited_by_id)})
    
    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data

#%%
# Question: How many papers do paper xxx cite?

random.seed(2031)
print("9-hop")
question = 'How many papers does paper "{paper_title}" cite?'
answer = "{num}"
generated_data = []

paper_ids = list(graph['paper_nodes'].keys())
random.shuffle(paper_ids)

for paper_id in paper_ids:
    paper_title = graph['paper_nodes'][paper_id]['features']['title']
    referred_by_id = graph['paper_nodes'][paper_id]['neighbors']['reference']
    if len(referred_by_id) == 0:
        continue
    generated_data.append({"paper_title": paper_title, "num": len(referred_by_id)})
    
    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
# Question: 4. How many papers did author xxx write?
random.seed(2033)
print("10-hop")
question = "How many papers did author {author_name} write?"
answer = "{num}"
generated_data = []

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id in author_ids:
    paper_ids = graph['author_nodes'][author_id]['neighbors']['paper']
    author_name = graph['author_nodes'][author_id]['features']['name']
    generated_data.append({"author_name": author_name,
                        "num": len(paper_ids)})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data

#Related Question How many papers does author xxx in xxx venue?
#Related Question How many papers does author xxx in xxx year?
#%% md
# ### medium question
# 1. Which is the most cited paper by author xxx?
# 2. Which venue did author xxx and author xxx collaborate most?
#%%
# Question: Which is the most cited paper by author xxx?
random.seed(2032)
print("11-hop")
question = "Which is the most cited paper by author {author_name}?"
answer = "{paper_title}"
generated_data = []

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id in author_ids:
    paper_ids = graph['author_nodes'][author_id]['neighbors']['paper']
    max_count = -1
    max_paper_id = None
    random.shuffle(paper_ids)
    for paper_id in paper_ids:
        
        cited_by_id = graph['paper_nodes'][paper_id]['neighbors']['cited_by']

        if len(cited_by_id) > max_count:
            max_count = len(cited_by_id)
            max_paper_id = paper_id
    
    paper_title = graph['paper_nodes'][max_paper_id]['features']['title']
    author_name = graph['author_nodes'][author_id]['features']['name']

    generated_data.append({"author_name": author_name,
                        "paper_title": paper_title})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data
#%%
# Question: 5. Which venue did author xxx and author xxx collaborate most?

random.seed(2034)
print("12-hop")
question = "Which venue did {author_name1} and {author_name2} collaborate most?"
answer = "{venue}"
generated_data = []

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

for author_id1 in author_ids:
    curr_author_ids = list(graph['author_nodes'].keys())
    random.shuffle(curr_author_ids)
    for author_id2 in curr_author_ids:

        if author_id1 == author_id2: 
            continue
        paper_ids1 = graph['author_nodes'][author_id1]['neighbors']['paper']
        paper_ids2 = graph['author_nodes'][author_id2]['neighbors']['paper']

        if len(set(paper_ids1) & set(paper_ids2)) < 1:
            continue

        count_per_venue = {}
        max_count = -1
        max_venue = None
        common_paper_ids = list(set(paper_ids1) & set(paper_ids2))

        for paper_id in common_paper_ids:
            venue = graph['paper_nodes'][paper_id]['neighbors']['venue'][0]
            if venue not in count_per_venue:
                count_per_venue[venue] = 0
            
            count_per_venue[venue] += 1
            if max_count < count_per_venue[venue]:
                max_count = count_per_venue[venue]
                max_venue = venue

        author_name1 = graph['author_nodes'][author_id1]['features']['name']
        author_name2 = graph['author_nodes'][author_id2]['features']['name']

        generated_data.append({"author_name1": author_name1,
                            "author_name2": author_name2,
                            "venue": graph['venue_nodes'][max_venue]['features']['name']})
        break

    if len(generated_data) == k:
            break

all_generated_data[(question, answer)] = generated_data

#Related Question: Which year did author xxx and author xxx collaborate most in?
#%% md
# Complex structure reasoning (medium)
# 1. How many people does author xxx need to know at least to know author xxx?
#%%
# Question: 1.  How many people does author xxx need to know at least to know author xxx?

random.seed(2035)
print("13-hop")
question = "How many people does author {author_name1} need to know at least to know author {author_name2}?"
answer = "{number}"
generated_data = []
max_hop_length = 5 # setting the maximum hop distance between two asked authors in the graph

author_ids = list(graph['author_nodes'].keys())
random.shuffle(author_ids)

def get_k_hop_neighbor(cur_author, hop, dist):
    
    queue = [cur_author]
    dist[cur_author] = 0
    
    while(len(queue)):
        cia = queue.pop(0)
        cur_papers = graph['author_nodes'][cia]['neighbors']['paper']
        cur_nids = []
        for pid in cur_papers:
            nids = graph['paper_nodes'][pid]['neighbors']['author']
            cur_nids.extend(nids)
        
        for cin in cur_nids:
            if cin in dist:
                continue
            dist[cin] = dist[cia] + 1
            if dist[cin] == hop:
                return cin
            queue.append(cin)
            
    return -1


for author_id in author_ids:
    cur_hop = random.randint(1, max_hop_length)
    neighbor = get_k_hop_neighbor(author_id, cur_hop, dict())
    if neighbor == -1:
        continue
    
    author_name1 = graph['author_nodes'][author_id]['features']['name']
    author_name2 = graph['author_nodes'][neighbor]['features']['name']

    generated_data.append({"author_name1": author_name1,
                        "author_name2": author_name2,
                        "number": cur_hop})
                               
    if len(generated_data) >= k:
            break

all_generated_data[(question, answer)] = generated_data

#Related Question: Which year did author xxx and author xxx collaborate most in?
#%% md
# Inductive reasoning (hard)
# 1. provide a paper recommendation for paper xxx 
#%%
## question (hard): provide a paper recommendation for paper xxx?

random.seed(2036)
print("14-hop")
# k = 3
question = "Which paper should be recommended to the reader of paper {paper1_title}? Please select from the candidate list {paper2_title}, {paper3_title}, {paper4_title}, {paper5_title}, {paper6_title}, {paper7_title}, {paper8_title}, {paper9_title}, {paper10_title}, {paper11_title}. Please answer the paper title rather than ID."
answer = "{paper_target_title}"
generated_data = []
for i in all_generated_data:
    print(i)
raw_pair = []
curr_set = set()

with open('PaperRecommendations.txt') as f:
    while True:
        line = f.readline()
        tmp = line.strip().split('\t')
        if tmp[0] in graph['paper_nodes'] and tmp[1] in graph['paper_nodes'] and tmp[0] not in curr_set and tmp[1] not in curr_set:
            raw_pair.append((tmp[0], tmp[1]))
            curr_set.add(tmp[0])
            curr_set.add(tmp[1])
        if len(raw_pair) == k * 5:
            break
print("broke")
random.shuffle(raw_pair)

paper_ids = list(graph['paper_nodes'].keys())
for pair in raw_pair:
    candidate_titles = []
    candidate_titles.append(graph['paper_nodes'][pair[1]]['features']['title'])
    random.shuffle(paper_ids)
    candidate_titles.append(graph['paper_nodes'][paper_ids[0]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[1]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[2]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[3]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[4]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[5]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[6]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[7]]['features']['title'])
    candidate_titles.append(graph['paper_nodes'][paper_ids[8]]['features']['title'])
    random.shuffle(candidate_titles)

    generated_data.append({"paper1_title": graph['paper_nodes'][pair[0]]['features']['title'],
                            "paper2_title": candidate_titles[0],
                            "paper3_title": candidate_titles[1],
                            "paper4_title": candidate_titles[2],
                            "paper5_title": candidate_titles[3],
                            "paper6_title": candidate_titles[4],
                            "paper7_title": candidate_titles[5],
                            "paper8_title": candidate_titles[6],
                            "paper9_title": candidate_titles[7],
                            "paper10_title": candidate_titles[8],
                            "paper11_title": candidate_titles[9],
                            "paper_target_title": graph['paper_nodes'][pair[1]]['features']['title']})

    if len(generated_data) == k:
        break

all_generated_data[(question, answer)] = generated_data

#%%
import json
import os

# Assuming all_generated_data is already populated from the previous code

# Ensure the directory exists
os.makedirs(domain, exist_ok=True)

# Convert dictionary keys to strings since JSON requires string keys
json_serializable_data = {}
for (question, answer), data in all_generated_data.items():
    json_serializable_data[f"{question}_{answer}"] = data

# Save to JSON file
with open(os.path.join(domain, 'data_og.json'), 'w') as f:
    json.dump(json_serializable_data, f, indent=2)

print('Saving file of #questions: ', len(json_serializable_data))
