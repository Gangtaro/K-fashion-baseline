import json
import os
import random


def split_dataset(input_json, input_csv, output_dir, val_ratio, random_seed):
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    # file_name에 prefix 디렉토리까지 포함 (CocoDataset 클래스를 사용하는 경우)
    # for image in images:
    #     image['file_name'] = '{}/{}'.format(image['file_name'][0], image['file_name'])

    image_ids = [x.get('id') for x in images]
    image_ids.sort()
    random.shuffle(image_ids)

    num_val = int(len(image_ids) * val_ratio)
    num_train = len(image_ids) - num_val

    image_ids_val, image_ids_train = set(image_ids[:num_val]), set(image_ids[num_val:])

    train_images = [x for x in images if x.get('id') in image_ids_train]
    val_images = [x for x in images if x.get('id') in image_ids_val]
    train_annotations = [x for x in annotations if x.get('image_id') in image_ids_train]
    val_annotations = [x for x in annotations if x.get('image_id') in image_ids_val]

    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories,
    }

    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories,
    }

    output_seed_dir = os.path.join(output_dir, f'seed{random_seed}')
    os.makedirs(output_seed_dir, exist_ok=True)
    output_train_json = os.path.join(output_seed_dir, 'train.json')
    output_val_json = os.path.join(output_seed_dir, 'val.json')
    output_train_csv = os.path.join(output_seed_dir, 'train.csv')
    output_val_csv = os.path.join(output_seed_dir, 'val.csv')

    print(f'write {output_train_json}')
    with open(output_train_json, 'w') as train_writer:
        json.dump(train_data, train_writer)

    print(f'write {output_val_json}')
    with open(output_val_json, 'w') as val_writer:
        json.dump(val_data, val_writer)

    print(f'write {output_train_csv}, {output_val_csv}')
    with open(input_csv, 'r') as csv_reader, \
            open(output_train_csv, 'w') as train_writer, \
            open(output_val_csv, 'w') as val_writer:
        train_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
        val_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
        for line in csv_reader:
            if line.startswith('ImageId'): continue
            image_id, encoded_pixels, height, width, category_id = line.strip().split(',')
            image_id = int(image_id)
            if image_id in image_ids_train:
                train_writer.write(line)
            elif image_id in image_ids_val:
                val_writer.write(line)
            else:
                raise ValueError(f'unknown image_id: {image_id}')
                
                
########### SPLIT DATASETS ######################
#################################################
split_dataset(input_json='../dataset/train.json',
              input_csv='../dataset/train.csv',
              output_dir='../dataset/',
              val_ratio=0.1,
              random_seed=23) # seed
              
              
########### TRAIN DATA SPLIT ###########              
import json

train_data = json.load(open('../dataset/seed23/train.json'))

print('training data')
print(f'images: {len(train_data["images"])}')
print(f'annotations: {len(train_data["annotations"])}')
print(f'categories: {len(train_data["categories"])}')


########### VALIDATION DATA SPLIT ###########
val_data = json.load(open('../dataset/seed13/val.json'))

print('validation data')
print(f'images: {len(val_data["images"])}')
print(f'annotations: {len(val_data["annotations"])}')
print(f'categories: {len(val_data["categories"])}')




#################################################
from .builder import DATASETS
from .coco import CocoDataset

# /mmdetection/mmdet/datasets/kfashion.py
# CocoDataset을 상속한 새로운 KFashionDataset을 정의
@DATASETS.register_module()
class KFashionDataset(CocoDataset):
    CLASSES = ('top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top', 'hood',
               'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants', 'coat', 'jacket',
               'jumper', 'padding jacket', 'best', 'kadigan', 'zip up', 'dress', 'jumpsuit')

    def load_annotations(self, ann_file):
        data_infos = super().load_annotations(ann_file)
        for x in data_infos:
            x['filename'] = '{}/{}'.format(x['filename'][0], x['filename'])
        return data_infos
        
    
# /mmdetection/mmdet/datasets/__init__.py
# __all__에 KFashionDataset을 추가합니다.
__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'DeepFashionDataset',
    'VOCDataset', 'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset',
    'LVISV1Dataset', 'GroupSampler', 'DistributedGroupSampler',
    'DistributedSampler', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'ClassBalancedDataset', 'WIDERFaceDataset', 'DATASETS', 'PIPELINES',
    'build_dataset', 'replace_ImageToTensor', 'KFashionDataset'
]



# /mmdetection/configs/_base_/default_runtime.py 혹은 사용하는 config 파일
# 사용하는 config 파일에 dataset_type을 KFashionDataset을 사용하도록 지정
dataset_type = 'KFashionDataset'
data_root = '/kfashion/dataset'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'seed23/train.json',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'seed23/val.json',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_pubilc.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline)
)


        



