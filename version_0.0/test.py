from ultralytics import YOLO
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pyvirtualcam
import os


model = YOLO('best.pt') 
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤：無法開啟實體攝像頭")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

print(f"解析度: {width}x{height}, FPS: {fps}")


with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:

    while True:
        if 'user_identity.npz' in os.listdir('.'):
            #判斷是否開啟成功            
            target_embedding = np.load('user_identity.npz', allow_pickle=True)
            while(not target_embedding):
                target_embedding = np.load('user_identity.npz', allow_pickle=True)
        else:
            target_embedding = {}
            
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        
        for r in results:
            for (x1, y1, x2, y2) in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                pad = 10
                
                face_crop = frame[max(0, y1-pad):min(height, y2+pad), 
                                  max(0, x1-pad):min(width, x2+pad)]
                
                if face_crop.size > 0:
                    face_crop_res = cv2.resize(face_crop, (240, 240))
                    faces = face_app.get(face_crop_res)
                    
                    label = "Unknown"
                    color = (0, 0, 255)
                    
                    if faces:
                        curr_emb = faces[0].embedding
                        score = -1
                        for name, t_emb in target_embedding.items():
                            s = compute_sim(curr_emb, t_emb)
                            if s > 0.4 and s > score:
                                score = s
                                label = name
                        
                        if score > 0.4:
                            color = (0, 255, 0)
                            label = f"{label} ({score:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. 發送畫面到虛擬鏡頭
        cam.send(frame)
        cam.sleep_until_next_frame()

        # 本地預覽（調試用，按 Q 退出）
        # cv2.imshow('Debug View', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

cap.release()
cv2.destroyAllWindows()