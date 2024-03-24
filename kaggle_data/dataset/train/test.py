import json

f = open("_annotations.coco.json")

data = json.load(f)
print("\n")
print("Keys", data.keys())

print('\n')
print("\nCategories:\n")
print(data['categories'])
print('\n')
# print(data['annotations'][:5])

print("ANNOTATIONS: TOP 5\n")
for _ in range(5):
    print(data['annotations'][_])
    print('\n')

print("\nIMAGE DATA: TOP 5\n")
for _ in range(5):
    print(data['images'][_])
    print('\n')

