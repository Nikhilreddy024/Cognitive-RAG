import json
import random

# Load the JSON objects line by line
with open("Physics/data.json", "r") as file:
    lines = file.readlines()

# Parse each line as a JSON object
data = {}
for line in lines:
    if line.strip():  # Skip empty lines
        obj = json.loads(line)
        data.update(obj)  # Combine all objects into one dictionary

# Define the range of keys to consider
start_key = 0
end_key = 100

# Loop through the range in batches of 10
for i in range(start_key, end_key + 1, 10):
    # Generate a batch of keys with the '$' prefix
    batch_keys = [f"$[{j}]" for j in range(i, min(i + 10, end_key + 1))]
    # Select 5 keys randomly from the batch
    keys_to_delete = random.sample(batch_keys, min(5, len(batch_keys)))
    # Delete these keys from the data
    for key in keys_to_delete:
        if key in data:
            del data[key]

# Save the modified data back to the file
with open("Physics/data.json", "w") as file:
    for key, value in data.items():
        json.dump({key: value}, file)
        file.write("\n")

print("Modified data saved to data.json.")
