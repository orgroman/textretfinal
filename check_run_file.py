import sys

def check_qids(path):
    qids = set()
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    qids.add(parts[0])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return

    print(f"Found {len(qids)} queries.")
    print(f"Sample qids: {list(qids)[:10]}")
    if '301' in qids:
        print("Qid 301 is present.")
    else:
        print("Qid 301 is NOT present.")

if __name__ == "__main__":
    check_qids("run_3_monot5.res")
