import os
import sys
import hashlib
import zipfile
from collections import defaultdict

# Import evaluation logic
from eval_and_verify import read_qrels, mean_ap, mean_ndcg, load_run

QRELS_PATH = "Files-20260104/qrels_50_Queries"

def check_format(path: str) -> bool:
    print(f"Checking format of {path}...")
    errors = 0
    seen = set()
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 6:
                print(f"  Line {line_num}: insufficient columns {len(parts)} (expected >=6)")
                errors += 1
                continue
                
            qid = parts[0]
            docid = parts[2]
            
            if (qid, docid) in seen:
                print(f"  Line {line_num}: Duplicate pair ({qid}, {docid})")
                errors += 1
            seen.add((qid, docid))
            
            if errors > 10:
                print("  Too many errors, stopping check.")
                return False
    
    if errors == 0:
        print("  Format OK.")
        return True
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python finalize_submission.py <run1> <run2> <run3> ...")
        sys.exit(1)
        
    run_files = sys.argv[1:]
    valid_runs = []
    
    # Check existence and format
    for rf in run_files:
        if not os.path.exists(rf):
            print(f"Error: {rf} does not exist.")
            continue
        if not check_format(rf):
            print(f"Error: {rf} has invalid format.")
            continue
        valid_runs.append(rf)
        
    if not valid_runs:
        print("No valid runs to package.")
        sys.exit(1)
        
    # Evaluate if possible
    if os.path.exists(QRELS_PATH):
        qrels = read_qrels(QRELS_PATH)
        print("\n--- Preliminary Evaluation (Judged Queries) ---")
        for rf in valid_runs:
            run = load_run(rf)
            run_judged = {q: d for q, d in run.items() if q in qrels}
            if run_judged:
                map_score = mean_ap(run_judged, qrels)
                print(f"{rf}: MAP={map_score:.4f} (on {len(run_judged)} queries)")
            else:
                print(f"{rf}: No judged queries found.")
    
    # Create Checksums
    print("\nGenerating checksums...")
    with open("checksums.md5", "w") as f_md5:
        for rf in valid_runs:
            with open(rf, "rb") as f_in:
                digest = hashlib.md5(f_in.read()).hexdigest()
                f_md5.write(f"{digest}  {rf}\n")
    print("checksums.md5 created.")
    
    # Zip
    zip_name = "submission.zip"
    print(f"\nCreating {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w') as zf:
        for rf in valid_runs:
            zf.write(rf)
        zf.write("checksums.md5")
        
    print("Done. Ready to submit!")

if __name__ == "__main__":
    main()
