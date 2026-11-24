"""Locate nested-Val differences offline without executing any model."""
import numpy as np
from files import GROUPS,ROOT,write_csv


def inspect():
    results=[]
    for group in GROUPS:
        subpath=ROOT/'runs'/group/('repaired_subset' if group=='O2M' else 'validation')/'epoch_20.npz'
        with np.load(subpath) as subset,np.load(ROOT/'full_val'/group/'epoch_20.npz') as full:
            sub_ids,sub_bound,sub_boxes=(subset[key] for key in ('ids','offsets','detections'))
            full_ids,full_bound,full_boxes=(full[key] for key in ('ids','offsets','detections'))
        lookup={image:slot for slot,image in enumerate(full_ids)}
        for sub_slot,image in enumerate(sub_ids):
            full_slot=lookup[image];left=sub_boxes[sub_bound[sub_slot]:sub_bound[sub_slot+1]];right=full_boxes[full_bound[full_slot]:full_bound[full_slot+1]]
            if not np.array_equal(left,right):
                results.append(dict(group=group,image_id=str(image),subset_slot_1based=sub_slot+1,full_slot_1based=full_slot+1,
                    subset_batch_size=min(4,len(sub_ids)-4*(sub_slot//4)),full_batch_size=min(4,len(full_ids)-4*(full_slot//4)),
                    subset_rows=len(left),full_rows=len(right),subset_max_score=float(left[:,4].max()) if len(left) else '',full_max_score=float(right[:,4].max()) if len(right) else ''))
    return results


def save_differences():
    results=inspect()
    if results:write_csv(ROOT/'full_val/nested_differences.csv',results)
    return results
