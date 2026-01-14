import os

INPUT = 'Files-20260104/queriesROBUST.txt'
OUTPUT = 'queries_judged.txt'
JUDGED_QIDS = [str(i) for i in range(301, 351)]

with open(INPUT, 'r') as fin, open(OUTPUT, 'w') as fout:
    for line in fin:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            if parts[0] in JUDGED_QIDS:
                fout.write(line)
print(f"Created {OUTPUT}")
