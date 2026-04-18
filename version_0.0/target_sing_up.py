from ultralytics import YOLO
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import glob



model = YOLO('best.pt') 
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

if 'user_identity.npz' in os.listdir('.'):
    print("已載入先前儲存的人臉向量")
    with np.load('user_identity.npz', allow_pickle=True) as data:
        if 'target_embedding' in data:
            target_embedding = data['target_embedding'].item()
        else:
            target_embedding = dict(data)
else:
    print("未找到先前儲存的人臉向量")
    target_embedding = {}

while True:

    for img_path in glob.glob('*.jpg'):
        results = model(img_path, verbose=False)
        name = img_path.split('.')[0]

        for r in results:
            frame = r.orig_img
            

            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                
                pad = 30
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)

                face_crop = frame[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, (240,240)) 

                faces = app.get(face_crop)
                
                if len(faces) == 0:
                    os.remove(img_path)
                    print(f"未檢測到臉部，已刪除 {img_path}")
                    pass
                else:
                    if(name in target_embedding):
                        pass
                    else:
                        target_embedding[name] = faces[0].embedding
                        
                        np.savez('user_identity.npz', **target_embedding)  
                        print(f"{name}人臉向量已儲存")

        