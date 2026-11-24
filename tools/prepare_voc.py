"""Freeze VOC train and val independently, before any model sees a label."""
import hashlib
import io
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from files import DATA, EVIDENCE, NAMES, SEED, digest, write_csv, write_json

ARCHIVE = DATA / 'VOCtrainval_06-Nov-2007.tar'
BASE = 'VOCdevkit/VOC2007/'
MD5 = 'c52e279531787c972589f7e41ab4ae64'


def parse_annotation(content, image_id):
    tree = ET.fromstring(content)
    w, h = (int(tree.findtext('size/' + s)) for s in ('width', 'height'))
    assert tree.findtext('filename') == image_id + '.jpg'
    objects = []
    for obj in tree.findall('object'):
        name = obj.findtext('name')
        x1, y1, x2, y2 = [int(obj.findtext('bndbox/' + v)) for v in ('xmin','ymin','xmax','ymax')]
        assert name in NAMES and 1 <= x1 <= x2 <= w and 1 <= y1 <= y2 <= h, (image_id, name, w, h, x1, y1, x2, y2)
        difficult, truncated = (int(obj.findtext(k, '0')) for k in ('difficult', 'truncated'))
        assert difficult in (0, 1) and truncated in (0, 1)
        # VOC uses 1-based inclusive corners. Convert to 0-based half-open edges.
        box = [x1 - 1, y1 - 1, x2, y2]
        objects.append(dict(cls=NAMES.index(name), xyxy=box, difficult=difficult, truncated=truncated))
    return dict(id=image_id, width=w, height=h, objects=objects)


def features(record):
    boxes = [o for o in record['objects'] if not o['difficult']]
    vec = np.zeros(24)
    for o in boxes:
        vec[o['cls']] = 1
        x, y, X, Y = o['xyxy']
        area = (X - x) * (Y - y) * (640 / max(record['width'], record['height'])) ** 2
        vec[20 if area < 32**2 else (21 if area < 96**2 else 22)] = 1
    vec[23] = len(boxes) >= 5
    return vec


def choose(records, n, rng):
    eligible = [r for r in records if any(not o['difficult'] for o in r['objects'])]
    features_all = np.stack([features(r) for r in eligible])
    expected = features_all.mean(0) * n
    best, best_score = None, float('inf')
    for _ in range(256):
        idx = rng.choice(len(eligible), n, replace=False)
        counts = features_all[idx].sum(0)
        if counts.min() == 0:
            continue
        score = np.mean((counts - expected) ** 2 / np.maximum(1, expected))
        if score < best_score:
            best, best_score = idx, score
    assert best is not None
    return sorted((eligible[i] for i in best), key=lambda r: r['id']), len(eligible), best_score, expected.tolist()


def case_ids(records):
    def regular(r):
        return [o for o in r['objects'] if not o['difficult']]
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])
    def overlap(r, same):
        objects = regular(r)
        scores = []
        for i, first in enumerate(objects):
            for second in objects[i + 1:]:
                if (first['cls'] == second['cls']) != same:
                    continue
                a, b = first['xyxy'], second['xyxy']
                inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                denom = area(a) + area(b) - inter if same else min(area(a), area(b))
                scores.append(inter / denom)
        return max(scores, default=0)
    single = [r for r in records if len(r['objects']) == 1 and not regular(r)[0]['truncated']]
    ordinary = min(single, key=lambda r: abs(area(regular(r)[0]['xyxy']) / (r['width']*r['height']) - .25))
    cases = {'ordinary': ordinary['id']}
    criteria = {
        'dense': lambda r: len(regular(r)),
        'small': lambda r: sum(area(o['xyxy'])*(640/max(r['width'],r['height']))**2 < 32**2 for o in regular(r)),
        'occlusion_candidate': lambda r: overlap(r, False),
        'duplicate_prone': lambda r: overlap(r, True),
    }
    for name, criterion in criteria.items():
        cases[name] = max((r for r in records if r['id'] not in cases.values()), key=criterion)['id']
    return cases


def main():
    assert digest(ARCHIVE, 'md5') == MD5, 'VOC archive does not match the torchvision reference'
    manifest = DATA / 'manifests'
    manifest.mkdir(exist_ok=True)
    hashes, class_rows, stats, chosen_sets = [], [], {}, {}
    with tarfile.open(ARCHIVE, 'r:') as tar:
        def extract(relative):
            return tar.extractfile(BASE + relative).read()
        for split, count, offset in [('train', 1000, 0), ('val', 250, 1)]:
            original = extract(f'ImageSets/Main/{split}.txt')
            (manifest / f'official_{split}.txt').write_bytes(original)
            ids = original.decode().split()
            records = [parse_annotation(extract(f'Annotations/{i}.xml'), i) for i in ids]
            chosen, eligible, score, target = choose(records, count, np.random.default_rng(SEED + offset))
            chosen_sets[split] = {r['id'] for r in chosen}
            out_ids = manifest / f'{split}.txt'
            out_ids.write_bytes((''.join(r['id'] + '\n' for r in chosen)).encode())
            write_json(manifest / f'{split}.json', chosen)
            for kind in ('JPEGImages', 'Annotations', 'labels'):
                (DATA / kind / split).mkdir(parents=True, exist_ok=True)
            for r in chosen:
                i, w, h = r['id'], r['width'], r['height']
                jpg, xml = extract(f'JPEGImages/{i}.jpg'), extract(f'Annotations/{i}.xml')
                with Image.open(io.BytesIO(jpg)) as im:
                    assert im.size == (w, h)
                    im.verify()
                (DATA / 'JPEGImages' / split / (i + '.jpg')).write_bytes(jpg)
                (DATA / 'Annotations' / split / (i + '.xml')).write_bytes(xml)
                lines = []
                for o in r['objects']:
                    if not o['difficult']:
                        x, y, X, Y = o['xyxy']
                        lines.append(f"{o['cls']} {(X+x)/(2*w):.9f} {(Y+y)/(2*h):.9f} {(X-x)/w:.9f} {(Y-y)/h:.9f}\n")
                label = ''.join(lines).encode()
                (DATA / 'labels' / split / (i + '.txt')).write_bytes(label)
                hashes.append(dict(split=split, image_id=i, jpeg_sha256=hashlib.sha256(jpg).hexdigest(),
                                   xml_sha256=hashlib.sha256(xml).hexdigest(), yolo_label_sha256=hashlib.sha256(label).hexdigest()))
            for c, name in enumerate(NAMES):
                def counts(pool):
                    regs = [[o for o in r['objects'] if o['cls']==c and not o['difficult']] for r in pool]
                    return sum(bool(v) for v in regs), sum(map(len, regs))
                ni, nb = counts(chosen)
                pi, pb = counts(records)
                diff = sum(o['cls']==c and o['difficult'] for r in chosen for o in r['objects'])
                class_rows.append(dict(split=split, class_id=c, name=name, regular_images=ni, regular_boxes=nb,
                                       difficult_boxes=diff, official_regular_images=pi, official_regular_boxes=pb))
            actual = np.stack([features(r) for r in chosen]).sum(0).astype(int).tolist()
            stats[split] = dict(official_images=len(ids), eligible_images=eligible, selected=count,
                                eligible_expected_features=target, selected_features=actual, sampling_score=score,
                                list_sha256=digest(out_ids), annotations_sha256=digest(manifest/f'{split}.json'),
                                regular_boxes=sum(x['regular_boxes'] for x in class_rows if x['split']==split),
                                difficult_boxes=sum(x['difficult_boxes'] for x in class_rows if x['split']==split))
            if split == 'val':
                write_json(manifest / 'cases.json', case_ids(chosen))
    assert not (chosen_sets['train'] & chosen_sets['val'])
    assert len({r['jpeg_sha256'] for r in hashes}) == 1250
    write_csv(manifest / 'file_hashes.csv', hashes)
    write_csv(manifest / 'class_counts.csv', class_rows)
    write_json(manifest / 'classes.json', NAMES)
    write_json(EVIDENCE / 'dataset.json', dict(archive=ARCHIVE.name, md5=MD5, sha256=digest(ARCHIVE), seed=SEED,
              source_url='https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar',
              train_val_id_overlap=0, train_val_identical_jpegs=0, splits=stats))
    print({s: (r['selected'], r['regular_boxes'], r['difficult_boxes']) for s,r in stats.items()})


if __name__ == '__main__':
    main()
