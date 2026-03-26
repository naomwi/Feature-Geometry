"""Download instructions and links for datasets used in the paper.

Datasets are too large to include in the repo. This script provides
download links and expected directory structure.

After downloading, your data/ directory should look like:

    data/
    ├── BSDS500/
    │   ├── images/test/          # 200 test images (.jpg)
    │   └── ground_truth/test/    # 200 .mat files
    ├── VOC2012/
    │   ├── JPEGImages/           # All images
    │   ├── SegmentationClass/    # Semantic masks (.png)
    │   └── ImageSets/Segmentation/val.txt
    ├── ADE20K/
    │   ├── images/validation/    # 2000 val images (.jpg)
    │   └── annotations/validation/  # 2000 masks (.png)
    └── COCO/
        ├── val2017/              # 5000 val images (.jpg)
        └── annotations/          # instances_val2017.json
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')

DATASETS = {
    'BSDS500': {
        'source': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html',
        'direct': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',
        'notes': 'Extract BSR/BSDS500/data/ -> data/BSDS500/',
    },
    'VOC2012': {
        'source': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/',
        'direct': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'notes': 'Extract VOCdevkit/VOC2012/ -> data/VOC2012/',
    },
    'ADE20K': {
        'source': 'https://groups.csail.mit.edu/vision/datasets/ADE20K/',
        'direct': 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
        'notes': 'Extract ADEChallengeData2016/ -> data/ADE20K/',
    },
    'COCO': {
        'source': 'https://cocodataset.org/#download',
        'direct': 'http://images.cocodataset.org/zips/val2017.zip + http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'notes': 'Extract val2017/ and annotations/ -> data/COCO/',
    },
}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Dataset Download Instructions")
    print("=" * 60)

    for name, info in DATASETS.items():
        path = os.path.join(DATA_DIR, name)
        exists = os.path.isdir(path)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"\n{name} [{status}]")
        print(f"  Source:   {info['source']}")
        print(f"  Download: {info['direct']}")
        print(f"  Notes:    {info['notes']}")
        print(f"  Path:     {path}")

    print("\n\nAfter downloading, place files as described above.")
    print("The experiments will look for data in: " + DATA_DIR)


if __name__ == '__main__':
    main()
