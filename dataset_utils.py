import os 
import shutil
import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def correct_dataset(dataset_path, remove_useless=True):
    for subject in os.listdir(dataset_path):
        patterns = [
            f"{subject}_color_0.png",
            f"{subject}_color_1.png",
            f"{subject}_depth_image_0.png",
            f"{subject}_depth_image_1.png",
            f"{subject}_ir_image_0.png",
            f"{subject}_ir_image_1.png",
            f"{subject}_points_0.ply",
            f"{subject}_points_1.ply",
            f"{subject}_smooth_depth_color_image_0.png",
            f"{subject}_smooth_depth_color_image_1.png",
            f"{subject}_transformed_point_0.ply",
            f"{subject}_transformed_point_1.ply"
        ]
        for emotion in os.listdir(os.path.join(dataset_path, subject)):
            img_path = os.path.join(dataset_path, subject, emotion)
            
            
            if os.path.exists(os.path.join(img_path, '00')):
                os.rmdir(os.path.join(img_path, '00'))
                
            for i, file in enumerate(os.listdir(os.path.join(img_path))):
                if file != patterns[i]:
                    os.rename(os.path.join(img_path, file), os.path.join(img_path, patterns[i]))
            
            if remove_useless:
                for i in range(2):
                    if os.path.exists(os.path.join(img_path, f'{subject}_ir_image_{i}.png')):
                        os.remove(os.path.join(img_path, f'{subject}_ir_image_{i}.png'))
                    if os.path.exists(os.path.join(img_path, f'{subject}_smooth_depth_color_image_{i}.png')):
                        os.remove(os.path.join(img_path, f'{subject}_smooth_depth_color_image_{i}.png'))
                    if os.path.exists(os.path.join(img_path, f'{subject}_points_{i}.ply')):
                        os.remove(os.path.join(img_path, f'{subject}_points_{i}.ply'))
                    if os.path.exists(os.path.join(img_path, f'{subject}_transformed_point_{i}.ply')):
                        os.remove(os.path.join(img_path, f'{subject}_transformed_point_{i}.ply'))
                    
            os.makedirs(os.path.join(img_path, '00'), exist_ok=True)
            

def clean_instructions_file(instructions_path, instructions_out_path):
    
    with open(instructions_path, 'r') as f:
        content = f.read()  
    with open(instructions_out_path, 'w') as f:
        f.write(content.lower())
    


def read_instuction_file(dataset_instructions_path):
    
        
        
            
    translate = {
        'anger': 'Colere',
        'disgust': 'Degout',
        'happiness': 'Joie',
        'neutrality': 'Neutre', 
        'fear': 'Peur', 
        'surprise': 'Surprise',
        'sadness': 'Tristesse',
        'no emotion' : 0
    }
    
    instructions = []
    with open(dataset_instructions_path, 'r') as f:
        for row in f:
            cols = [col.strip() for col in row.split(';')]
            subject, emotion, image, folder = cols
            instructions.append({
                'subject': 'K' + str(int(subject[2:])).zfill(3),
                'emotion': translate.get(emotion, 0),
                'image': image,
                'folder': translate.get(folder, 1)
            })
    return instructions



def clean_dataset(dataset_path, instructions):
    for row in instructions:
        sub_path = os.path.join(dataset_path, row['subject'])

        if row['emotion'] == 0:
            for emotion in os.listdir(sub_path):
                path = os.path.join(sub_path, emotion)
                for img in os.listdir(path)[1:]:
                    for i in range(2):
                        color_path = row['subject'] + '_color_' + str(i) + '.png'
                        depth_path = row['subject'] + '_depth_image_' + str(i) + '.png'
                        if os.path.exists(os.path.join(path, color_path)):
                            shutil.move(os.path.join(path, color_path), os.path.join(path,'00',color_path))
                        if os.path.exists(os.path.join(path, depth_path)):
                            shutil.move(os.path.join(path, depth_path), os.path.join(path,'00',depth_path))

        
        else :
            
            path = os.path.join(sub_path, row['emotion'])
            color_path = row['subject'] + '_color_' + str(int(row['image'])-1) + '.png'
            depth_path = row['subject'] + '_depth_image_' + str(int(row['image'])-1) + '.png'
            
            if row['folder'] == 0:
                if os.path.exists(os.path.join(path, color_path)):
                    os.remove(os.path.join(path, color_path))
                if os.path.exists(os.path.join(path, depth_path)):
                    os.remove(os.path.join(path, depth_path))
            
            if row['folder'] == 1:
                if os.path.exists(os.path.join(path, color_path)):
                    shutil.move(os.path.join(path, color_path), os.path.join(path,'00',color_path))
                if os.path.exists(os.path.join(path, depth_path)):
                    shutil.move(os.path.join(path, depth_path), os.path.join(path,'00',depth_path))
            
            else:
                new_color_path = row['subject'] + '_color_' + str(int(row['image'])-1) + f"_{row['emotion']}_moved.png"
                new_depth_path = row['subject'] + '_depth_image_' + str(int(row['image'])-1) + f"_{row['emotion']}_moved.png"
                if os.path.exists(os.path.join(path, color_path)):
                    shutil.move(os.path.join(path, color_path), os.path.join(sub_path, row['folder'], new_color_path))
                if os.path.exists(os.path.join(path, depth_path)):
                    shutil.move(os.path.join(path, depth_path), os.path.join(sub_path, row['folder'], new_depth_path))


def count_images(dataset_path):
    nb_img = 0
    for subject in os.listdir(dataset_path):
        for emotion in os.listdir(os.path.join(dataset_path, subject)):
            for img_name in os.listdir(os.path.join(dataset_path, subject, emotion))[1:]:
                if 'color' in img_name:
                    nb_img += 1 
    return nb_img


def load_dataset(dataset_path):
    nb_img = count_images(dataset_path)
    
    img_rgb = np.empty(shape=(nb_img, 1080, 1920, 3), dtype=np.float32)
    img_depth = np.empty(shape=(nb_img, 288, 320, 3), dtype=np.float32)
    targets = []

    i = 0
    j = 0
    for subject in os.listdir(dataset_path):
        for emotion in os.listdir(os.path.join(dataset_path, subject)):
            for img_name in os.listdir(os.path.join(dataset_path, subject, emotion))[1:]:
                img = np.array(plt.imread(os.path.join(dataset_path, subject, emotion, img_name)))
                if 'color' in img_name:
                    img_rgb[i] = img
                    i += 1
                    targets.append(emotion) 
                if 'depth' in img_name:
                    img_depth[j] = img
                    j += 1
    
    return img_rgb, img_depth, np.array(targets)



def fix_size(fn, desired_w, desired_h, fill_color=(0, 0, 0, 255)):
    im = cv2.imread(fn, cv2.IMREAD_COLOR_RGB)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Obtenir les dimensions de l'image
    y, x, _ = im.shape
    
    ratio = x / y
    desired_ratio = desired_w / desired_h

    w = max(desired_w, x)
    h = int(w / desired_ratio)
    if h < y:
        h = y
        w = int(h * desired_ratio)

    # Créer une nouvelle image remplie de la couleur spécifiée
    new_im = np.full((h, w, 3), fill_color[:3], dtype=np.uint8)

    # Coller l'image d'origine au centre de la nouvelle image
    x_offset = (w - x) // 2
    y_offset = (h - y) // 2
    new_im[y_offset:y_offset+y, x_offset:x_offset+x, :] = im

    # Redimensionner l'image à la taille désirée
    new_im = cv2.resize(new_im, (desired_w, desired_h))

    return new_im


def resize_with_padding(image, target_size=100, pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=pad_color)
    
    return padded_image

def load_data(data_path):
    nb_imgs = 0
    for emotion in os.listdir(data_path):
        nb_imgs += len(os.listdir(os.path.join(data_path, emotion, 'rgb')))
    
    img_rgb = np.empty(shape=(nb_imgs, 100, 100, 3), dtype=np.float32)
    img_depth = np.empty(shape=(nb_imgs, 100, 100, 3), dtype=np.float32)
    labels = []
    i = 0
    for emotion in sorted(os.listdir(data_path)):
        rgb_path = os.path.join(data_path, emotion, 'rgb')
        depth_path = os.path.join(data_path, emotion, 'depth')
        for rgb_img, depth_img in zip(sorted(os.listdir(rgb_path)), sorted(os.listdir(depth_path))):
            img_1 = fix_size(os.path.join(data_path, emotion, 'rgb', rgb_img), desired_h=100, desired_w=100) / 255.
            img_rgb[i] = img_1
            img_2 = fix_size(os.path.join(data_path, emotion, 'depth', depth_img), desired_h=100, desired_w=100) / 255.
            img_depth[i] = img_2
            labels.append(emotion)
            i += 1
    
            
    return img_rgb, img_depth, np.array(labels)

def check_img(img_rgb, img_depth, labels):
    check_path = './img_check'
    rgb_path = os.path.join(check_path, 'rgb')
    depth_path = os.path.join(check_path, 'depth')
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    
    for i, img in enumerate(img_rgb):
        plt.imsave(os.path.join(rgb_path, 'rgb_' + f'{i}'.zfill(4) + f'_{labels[i]}'+'.png'), img)
            

    for i, img in enumerate(img_depth):
        plt.imsave(os.path.join(depth_path, 'depth_' + f'{i}'.zfill(4) + f'_{labels[i]}'+'.png'), img)

def prepare_data(imgs, labels, n_splits=10):
    le = LabelEncoder()
    kf = KFold(n_splits, shuffle=True, random_state=42)
    k_fold = []
    
    for fold, (train_ind, test_ind) in enumerate(kf.split(imgs)):
        X_train = imgs[train_ind]
        X_test = imgs[test_ind]
        y_train = labels[train_ind]
        y_test = labels[test_ind]
        
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        y_train = to_categorical(y_train, 7)
        y_test = to_categorical(y_test, 7)

        
        k_fold.append((
            X_train,
            y_train,
            X_test,
            y_test
        ))
        
    return k_fold, le

def load_and_concat(data_path):
    nb_imgs = 0
    for emotion in os.listdir(data_path):
        nb_imgs += len(os.listdir(os.path.join(data_path, emotion, 'rgb')))
        
    i = 0
    imgs = np.empty(shape=(nb_imgs, 100, 100, 3), dtype=np.float32)
    labels = []
    for emotion in sorted(os.listdir(data_path)):
            rgb_path = os.path.join(data_path, emotion, 'rgb')
            depth_path = os.path.join(data_path, emotion, 'depth')
            for rgb_img, depth_img in zip(sorted(os.listdir(rgb_path)), sorted(os.listdir(depth_path))):
                img_rgb = cv2.imread(os.path.join(data_path, emotion, 'rgb', rgb_img), cv2.IMREAD_COLOR_RGB) 
                img_rgb = img_rgb / 255.0
                img_depth = cv2.imread(os.path.join(data_path, emotion, 'depth', depth_img), cv2.IMREAD_COLOR_RGB) 
                img_depth = img_depth / 255.0
                concat = cv2.hconcat([img_rgb, img_depth])
                imgs[i] = resize_with_padding(concat, 100)
                labels.append(emotion)
                i += 1
    return imgs, np.array(labels)