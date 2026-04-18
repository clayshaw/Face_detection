from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import time

class detect:
    def __init__(self):

        # self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        # self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.app = FaceAnalysis(name='buffalo_sc',allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        
        self.cap = cv2.VideoCapture(0)
        
        self.target_embedding = self.load_identity()
        

        self.ret, self.frame = self.cap.read()
        self.ticks = 0
        self.last_results = [] 
        self.recognition_interval = 3

    def load_identity(self):
        if 'user_identity.npy' in os.listdir('.'):
            print("已載入先前儲存的人臉向量")
            data = np.load('user_identity.npy', allow_pickle=True)
            try:
                return data.item()
            except:
                return data['arr_0'].item() if 'arr_0' in data else {}
        else:
            print("未找到先前儲存的人臉向量")
            return {}

    def compute_sim(self, feat1, feat2):
        # 餘弦相似度計算
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def sing_up(self, name):
        """註冊新臉孔"""
        if self.frame is not None:
            faces = self.app.get(self.frame)
            if faces:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                self.target_embedding[name] = faces[0].embedding
                np.save('user_identity', self.target_embedding)
                print(f"{name} 人臉向量已儲存")
            else:
                print("未偵測到人臉，無法註冊")

    def run(self):
        while True:
            begin_time = time.time()
            self.ret, draw_frame = self.cap.read()
            if not self.ret:
                break

            if self.ticks % self.recognition_interval == 0:
                # 執行偵測與特徵提取
                small_frame = cv2.resize(draw_frame, (0, 0), fx=0.5, fy=0.5)
                faces = self.app.get(small_frame)
                
                new_results = []
                for face in faces:
                    # 取得座標
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    
                    label = "Unknown"
                    color = (0, 0, 255) # 紅色
                    curr_emb = face.embedding
                    
                    score = -1
                    best_name = None
                    for name, t_emb in self.target_embedding.items():
                        s = self.compute_sim(curr_emb, t_emb)
                        if s > 0.4 and s > score:
                            score = s
                            best_name = name
                    
                    if score > 0.4:
                        label = f"{best_name} ({score:.2f})"
                        color = (0, 255, 0) # 綠色
                    
                    new_results.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': label,
                        'color': color
                    })
                self.last_results = new_results

            for res in self.last_results:
                x1, y1, x2, y2 = res['bbox']
                cv2.rectangle(draw_frame, (x1*2, y1*2), (x2*2, y2*2), res['color'], 2)
                cv2.putText(draw_frame, res['label'], (x1*2, y1*2 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, res['color'], 2)

            self.frame = draw_frame
            self.ticks += 1
            
            # 控制 ticks 大小避免無限增長
            if self.ticks > 1000: self.ticks = 0

            end_time = time.time()
            print(f"每幀處理時間: {(end_time - begin_time)*1000:.2f} ms")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()