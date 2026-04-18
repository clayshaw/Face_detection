from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import time


class detector:
    def __init__(self):
        
        #用GPU
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        
        
        #用CPU
        # self.app = FaceAnalysis(name='buffalo_sc',allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        # self.app.prepare(ctx_id=-1, det_size=(320, 320))
        if self.app is None:
            raise Exception("無法初始化人臉分析模型")

        self.frame = None
        self.average_execution_time = 0
        
        self.target_embedding = self.load_identity()


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

    def sing_up(self, name, frame):
        """註冊新臉孔"""
        if frame is not None:
            faces = self.app.get(frame)
            if faces:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                for face in self.target_embedding.values():
                    score = self.compute_sim(faces[0].embedding, face)
                    if score > 0.5 or name in self.target_embedding:  # 設定相似度閾值
                        print("此人臉已存在，無法註冊")
                        return
                self.target_embedding[name] = faces[0].embedding
                   
                np.save('user_identity', self.target_embedding)
                print(f"{name} 人臉向量已儲存")
            else:
                print("未偵測到人臉，無法註冊")

    def detect(self, frame):
        begin_time = time.time()
        best_name = "Unknown"
        best_score = 0.0
        color = (0, 0, 255) 
        tmp_frame = frame.copy() if frame is not None else None

        if tmp_frame is not None:
            faces = self.app.get(tmp_frame)
            
            if faces:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                best_face = faces[0]
                x1, y1, x2, y2 = map(int, best_face.bbox)

                for name, embedding in self.target_embedding.items():
                    score = self.compute_sim(best_face.embedding, embedding)
                    if score > best_score:
                        best_score = score
                        best_name = name 

                if best_score > 0.5: 
                    color = (0, 255, 0)
                else:
                    best_name = "Unknown"


                cv2.rectangle(tmp_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(tmp_frame, f"{best_name} ({best_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        self.frame = tmp_frame
        end_time = time.time()
        execution_time = end_time - begin_time
        alpha = 0.1
        self.average_execution_time = (alpha * execution_time) + (1 - alpha) * self.average_execution_time


    def __del__(self):
        print("EMA執行時間指數: {:.4f} ".format(self.average_execution_time*1000))
        self.app = None