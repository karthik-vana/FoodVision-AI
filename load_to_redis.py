import json
import redis

# Step 1: Connect to Redis server
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Step 2: Load the JSON file
with open("nutrition_data.json", "r") as f:
    data = json.load(f)

# Step 3: Store each food item in Redis
for food, nutrition in data.items():
    r.set(food, json.dumps(nutrition))  # Store as JSON string

print(f" Stored {len(data)} food items in Redis.")

# Step 4: Verify by reading one item back
test = r.get("apple_pie")
if test:
    print("\n Verification:")
    print("apple_pie:", json.loads(test))
else:
    print(" 'apple_pie' not found in Redis.")
    
