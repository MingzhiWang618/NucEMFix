import json
import os

def summarize_results(json_paths):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print(f"")
    print("-" * 40)

    for path in json_paths:
        if not os.path.exists(path):
            print(f"")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            

        summary = data.get("summary", {})
        tp = summary.get("tp") or summary.get("total_tp", 0)
        fp = summary.get("fp") or summary.get("total_fp", 0)
        fn = summary.get("fn") or summary.get("total_fn", 0)
        
        print(f"File: {os.path.basename(path):<20} | TP: {tp:>5}, FP: {fp:>5}, FN: {fn:>5}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Precision = TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    # Recall = TP / (TP + FN)
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # F1 = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Accuracy (CSI) = TP / (TP + FP + FN)
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    print("-" * 40)
    print(f"{'Final Combined Metrics':^40}")
    print("-" * 40)
    print(f"{'Total TP':<20} : {total_tp}")
    print(f"{'Total FP':<20} : {total_fp}")
    print(f"{'Total FN':<20} : {total_fn}")
    print("-" * 40)
    print(f"{'Precision':<20} : {precision:.4f}")
    print(f"{'Recall':<20} : {recall:.4f}")
    print(f"{'F1-Score':<20} : {f1:.4f}")
    print(f"{'Accuracy (Acc)':<20} : {accuracy:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize batch correction results from JSON files")
    parser.add_argument("json_files", nargs="+", help="Paths to result JSON files")
    args = parser.parse_args()
    summarize_results(args.json_files)