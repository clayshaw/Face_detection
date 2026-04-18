import tkinter as tk
from detect import detect
from PIL import Image, ImageTk
import cv2
import threading


window = tk.Tk()
window.title("YOLOv8 Object Detection")
window.geometry("700x600")
window.resizable(False, False)
my_detector = detect()

video_label = tk.Label(window, text="Video Stream")
video_label.pack(pady=20)

input_frame = tk.Frame(window)
input_frame.pack(pady=10)

text_input = tk.Entry(input_frame)
text_input.pack(side=tk.LEFT, padx=5)

btn = tk.Button(input_frame, text="take photo", command=lambda: my_detector.sing_up(text_input.get()))
btn.pack(side=tk.LEFT, padx=5)


def update_gui():
    if hasattr(my_detector, 'frame'):
        try:
            img = cv2.cvtColor(my_detector.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            video_label.imgtk = imgtk 
            video_label.configure(image=imgtk)
        except Exception as e:
            print(f"更新畫面錯誤: {e}")

    window.after(30, update_gui)

threading.Thread(target=my_detector.run, daemon=True).start()




update_gui()
window.mainloop()