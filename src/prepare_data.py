import glob
from argparse import ArgumentParser
import  os
import yaml
import random
import shutil

def prepare_data(config):
    input = config.get('input_path')
    output = config.get('output_path')
    train_rate=config.get('training_rate')
    val_rate = config.get('val_rate')
    test_rate = config.get('test_rate')

    train_ratio = train_rate/(train_rate+val_rate+test_rate)
    val_ratio = val_rate/(train_rate+val_rate+test_rate)
    test_ratio = test_rate/(train_rate+val_rate+test_rate)

    folders = [f for f in os.listdir(input) if os.path.isdir(os.path.join(input, f))]

    # 確保輸出資料夾存在
    for split in ['train', 'val', 'test']:
        for category in folders:
            os.makedirs(os.path.join(output, split, category), exist_ok=True)

    for category in folders:
        category_dir = os.path.join(input, category)
        images = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        total = len(images)
        train_count = int(total*train_ratio)
        val_count = int(total*val_ratio)
        test_count = int(total*test_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:train_count+val_count]
        test_images = images[train_count+val_count:]
        for img in train_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(output, 'train', category, img))

        for img in val_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(output, 'val', category, img))

        for img in test_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(output, 'test', category, img))

        print("Total images: {} process done".format(total))



def get_config(config):
    with open(config,'r') as stream:
        return yaml.load(stream,Loader=yaml.Loader)

if __name__ == '__main__':
    """
        指令yaml 檔
        將資料夾以config 中的比例分割為train, val, test 三個區塊
        ex:
        python prepare_data.py --config-file config.yaml
    
    """
    parser = ArgumentParser()
    parser.add_argument('--config',type=str,default="config.yaml",required=True,dest='config')
    args = parser.parse_args()

    config = get_config(args.config)
    prepare_data(config)