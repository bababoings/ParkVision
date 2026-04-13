import os
import json
import random
import cv2
from tqdm import tqdm

def process_coco_subset(split_name, json_path, img_dir, output_base, limit_per_class=10000):
    print(f"Processing COCO annotations for '{split_name}'...")
    
    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    # Map category names to output folders
    cat_map = {}
    for cat in coco['categories']:
        if cat['name'] == 'space-empty':
            cat_map[cat['id']] = 'empty'
        elif cat['name'] == 'space-occupied':
            cat_map[cat['id']] = 'occupied'
            
    # Map image IDs to file names
    img_map = {img['id']: img['file_name'] for img in coco['images']}
    
    # Collect annotations per category
    data_by_class = {'empty': [], 'occupied': []}
    for ann in coco['annotations']:
        cat_id = ann.get('category_id')
        if cat_id in cat_map:
            cls_name = cat_map[cat_id]
            data_by_class[cls_name].append({
                'img_id': ann['image_id'],
                'bbox': ann['bbox'] # [x,y,w,h]
            })
            
    # Sample and extract
    for cls_name, items in data_by_class.items():
        # Llevamos el registro aleatorio para que haya variacion
        random.seed(42) # Reproducibilidad
        random.shuffle(items)
        if limit_per_class > 0:
            items = items[:limit_per_class]
            
        out_dir = os.path.join(output_base, cls_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"  Extracting {len(items)} samples for '{cls_name}'...")
        count = 0
        
        # Optimize by parsing image-by-image
        items_by_image = {}
        for item in items:
            items_by_image.setdefault(item['img_id'], []).append(item['bbox'])
            
        for img_id, bboxes in tqdm(items_by_image.items(), desc=f"{cls_name} images"):
            img_path = os.path.join(img_dir, img_map[img_id])
            if not os.path.exists(img_path):
                continue
                
            img_obj = cv2.imread(img_path)
            if img_obj is None:
                continue
                
            h_img, w_img = img_obj.shape[:2]
            
            for bbox in bboxes:
                # COCO format: x_min, y_min, width, height
                x, y, w, h = map(int, bbox)
                
                # Sanity checking
                x, y = max(0, x), max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                if w <= 0 or h <= 0:
                    continue
                    
                crop = img_obj[y:y+h, x:x+w]
                crop_resized = cv2.resize(crop, (96, 96))
                
                out_path = os.path.join(out_dir, f"pklot_{split_name}_{cls_name}_{count}.jpg")
                cv2.imwrite(out_path, crop_resized)
                count += 1

if __name__ == '__main__':
    PKLOT_BASE = r"D:\Pklot_dataset"
    OUTPUT_TRAIN = r"train_data\train"
    OUTPUT_VAL = r"train_data\test"
    
    # Process Train (10k per class)
    process_coco_subset(
        'train',
        os.path.join(PKLOT_BASE, r"train\_annotations.coco.json"),
        os.path.join(PKLOT_BASE, "train"),
        OUTPUT_TRAIN,
        limit_per_class=10000
    )
    
    # Process Valid (2k per class)
    process_coco_subset(
        'valid',
        os.path.join(PKLOT_BASE, r"valid\_annotations.coco.json"),
        os.path.join(PKLOT_BASE, "valid"),
        OUTPUT_VAL,
        limit_per_class=2000
    )
    print("Done! ROIs successfully extracted to train_data/")
