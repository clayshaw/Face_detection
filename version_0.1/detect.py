from ultralytics import YOLO
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import time




class detect:
    def __init__(self):
        self.model = YOLO('best.pt') 

        # gpu執行 原l版辨識 + 偵測
        # self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        # self.app.prepare(ctx_id=0, det_size=(640, 640))

        # cpu執行 m版辨識 + sc版偵測
        self.app = FaceAnalysis(name='buffalo_sc',allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        

        self.cap = cv2.VideoCapture(0)
        if 'user_identity.npy' in os.listdir('.'):
            print("已載入先前儲存的人臉向量")
            data = np.load('user_identity.npy', allow_pickle=True)
            try:
                self.target_embedding = data.item()
            except:
                self.target_embedding = data['arr_0'].item() if 'arr_0' in data else {}
        else:
            print("未找到先前儲存的人臉向量")
            self.target_embedding = {}
        self.ret, self.frame = self.cap.read()
            

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def sing_up(self,name):
        results = self.model(self.frame, verbose=False)

        for r in results:
            inner_frame = r.orig_img
            

            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                
                pad = 30
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(inner_frame.shape[1], x2 + pad)
                y2 = min(inner_frame.shape[0], y2 + pad)

                face_crop = inner_frame[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, (240,240)) 

                faces = self.app.get(face_crop)
                
                if name not in self.target_embedding:
                    curr_emb = faces[0].embedding
                    score = -1
                    for name, t_emb in self.target_embedding.items():
                        s = self.compute_sim(curr_emb, t_emb)
                        if s > 0.4 and s > score:
                            score = s
                            label = name
                    
                    if score > 0.4:
                        print(f"已存在相似人臉: {label} (相似度: {score:.2f})")
                    else:
                        self.target_embedding[name] = faces[0].embedding
                        np.save('user_identity', self.target_embedding)  
                        print(f"{name} 人臉向量已儲存")



    def compute_sim(self, feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def run(self):
        while True:
            begin_time = time.time()

            self.ret, draw_frame = self.cap.read()
            if not self.ret:
                break
            results = self.model(draw_frame, verbose=False)
            
            for r in results:
                for (x1, y1, x2, y2) in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])


                    pad = 10
                    
                    face_crop = draw_frame[max(0, y1-pad):min(draw_frame.shape[0], y2+pad), 
                                        max(0, x1-pad):min(draw_frame.shape[1], x2+pad)]
                    
                    if face_crop.size > 0:
                        small_frame = cv2.resize(draw_frame, (0, 0), fx=0.5, fy=0.5)
                        faces = self.app.get(small_frame)
                        
                        label = "Unknown"
                        color = (0, 0, 255)
                        
                        if faces:
                            curr_emb = faces[0].embedding
                            score = -1
                            for name, t_emb in self.target_embedding.items():
                                s = self.compute_sim(curr_emb, t_emb)
                                if s > 0.4 and s > score:
                                    score = s
                                    label = name
                            
                            if score > 0.4:
                                color = (0, 255, 0)
                                label = f"{label} ({score:.2f})"

                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(draw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            self.frame = draw_frame
            # cv2.imshow('YOLOv8 Object Detection', self.frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            end_time = time.time()
            print(f"每幀處理時間: {(end_time - begin_time)*1000:.2f} ms")

if __name__ == "__main__":
    detector = detect()
    detector.run()