import os
import sys
import json
import boto3
import shutil
from pathlib import Path


def normalize_coords(x1, y1, w, h, image_w, image_h):
    return ["%.6f" % ((2*x1 + w)/(2*image_w)) , "%.6f" % ((2*y1 + h)/(2*image_h)), "%.6f" % (w/image_w), "%.6f" % (h/image_h)]


def convert_annotations(manifest_jsonl, folder, dict_key):
    for manifest_jsonl_line in manifest_jsonl:
        height = manifest_jsonl_line[dict_key]['image_size'][0]['height']
        width = manifest_jsonl_line[dict_key]['image_size'][0]['width']
        writer = ''
        image_uri = manifest_jsonl_line['source-ref']
        filename = os.path.basename(image_uri)
        ext, filename = filename[::-1].split('.', 1)
        ext = ext[::-1]
        filename = filename[::-1]
        for annotation in manifest_jsonl_line[dict_key]['annotations']:
            normalized_points = normalize_coords(annotation['left'], annotation['top'], annotation['width'], annotation['height'], width, height)
            # print(normalized_points)
            writer += f'{index} {normalized_points[0]} {normalized_points[1]} {normalized_points[2]} {normalized_points[3]} \n'
        with open(f"custom_dataset/labels/{folder}/{filename}.txt", 'w') as f:
            f.write(writer)
        bucket, key = image_uri[5:].split('/', 1)
        s3.Bucket(bucket).download_file(key, f"custom_dataset/images/{folder}/{filename}.{ext}")


classes = ["dog", "person", "cat", "tv", "car", "meatballs", "marinara sauce", "tomato soup", "chicken noodle soup", "french onion soup", "chicken breast", "ribs", "pulled pork", "hamburger", "cavity"]

s3 = boto3.resource('s3')
smgt_client = boto3.client('sagemaker', region_name='us-west-2')
queue_name = sys.argv[1]  #-- "V3-Cup-BatchP-BBox-Inhouse-chain"
label = sys.argv[2]
train_samples = int(sys.argv[3])
test_val = int(train_samples * 0.10)
manifest_jsonl = []
if label not in classes:
    classes.append(label)
    index = 15
else:
    index = classes.index(label)

#refresh folders
if os.path.exists('custom_dataset'):
    shutil.rmtree('custom_dataset')
cwd = os.getcwd()
Path(cwd + "/custom_dataset/images/test").mkdir(parents=True, exist_ok=True)
Path(cwd + "/custom_dataset/images/train").mkdir(parents=True, exist_ok=True)
Path(cwd + "/custom_dataset/images/val").mkdir(parents=True, exist_ok=True)
Path(cwd + "/custom_dataset/labels/test").mkdir(parents=True, exist_ok=True)
with open(cwd + "/custom_dataset/labels/test/classes.txt", 'w') as f:
    for _class in classes:
        f.write(_class)
        f.write('\n')
Path(cwd + "/custom_dataset/labels/train").mkdir(parents=True, exist_ok=True)
with open(cwd + "/custom_dataset/labels/train/classes.txt", 'w') as f:
    for _class in classes:
        f.write(_class)
        f.write('\n')
Path(cwd + "/custom_dataset/labels/val").mkdir(parents=True, exist_ok=True)
with open(cwd + "/custom_dataset/labels/val/classes.txt", 'w') as f:
    for _class in classes:
        f.write(_class)
        f.write('\n')


response = smgt_client.describe_labeling_job(
                LabelingJobName=queue_name
            )

if response["LabelingJobStatus"] == "Completed":
    output_manifest_path = response["OutputConfig"]["S3OutputPath"] + f"{queue_name}/manifests/output/output.manifest"
    bucket, key = output_manifest_path[5:].split('/', 1)
    manifest_obj = s3.Object(bucket, key)
    manifest_lines = manifest_obj.get()['Body'].read().decode().splitlines()
    for manifest_line in manifest_lines:
        manifest_data = json.loads(manifest_line)
        manifest_jsonl.append(manifest_data)
    print(f'Finished parsing manifest: {queue_name}')
else:
    print('Job not completed yet')
    sys.exit()

key = [keys for keys in manifest_jsonl[0].keys() if not keys.endswith('-metadata') and not keys == 'source-ref' ][0]
print(key)

image_count = len(manifest_jsonl)

training_set = manifest_jsonl[:train_samples]
convert_annotations(training_set, "train", key)
val_set = manifest_jsonl[train_samples :train_samples + test_val]
convert_annotations(val_set, "val", key)
test_set = manifest_jsonl[train_samples + test_val:train_samples + (2 * test_val)]
convert_annotations(test_set, "test", key)
writer = ''
writer += f'train: {cwd}/custom_dataset/images/train\n'
writer += f'val: {cwd}/custom_dataset/images/val\n'
writer += f'test: {cwd}/custom_dataset/images/test\n'
writer += "is_coco: False\n"
writer += f"nc: {len(classes)}\n"
writer += f"names: {str(classes)}"

with open('data/dataset.yaml', 'w') as f:
    f.write(writer)