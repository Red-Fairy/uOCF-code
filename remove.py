import os

root = '/svl/u/redfairy/datasets/room-real/chairs/train-2-4obj'

files = os.listdir(root)

count = 0

for file in files:
    if 'pfm' in file and 'mask' in file:
        # delete
        # count += 1
        os.remove(os.path.join(root, file))

print(count)