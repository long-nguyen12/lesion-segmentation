from glob import glob
print(glob(f"./data/K-Fold-Validation/*/", recursive=True))
for i, ds in enumerate(glob(f"/data/K-Fold-Validation/*/", recursive=True)):
    print(ds)