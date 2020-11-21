import tkinter as tk
# import tensorflow as tf
#import numpy as np
#import cv2

def classify():
    path = entry_path.get()
    categories = ["Dog", "Cat"]
    def prepare(path):
        image_size = 70
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
        new_array = cv2.resize(img_array, (image_size, image_size))  
        return np.array(new_array.reshape(-1, image_size, image_size, 1))
    ########################################
    model = tf.keras.models.load_model("3-conv-64-nodes-2-dense-1589461361")
    ########################################
    prediction = model.predict([prepare(path)]) 
    result = categories[int(prediction[0][0])]
    label_class["text"] = str(result)
    return 



root = tk.Tk()
root.title("cnn binary classifier")

for i in range(2):
    root.columnconfigure(i, weight=1, minsize=75)
    root.rowconfigure(i, weight=1, minsize=50)

frame_path = tk.Frame(master=root)
label_path = tk.Label(master=frame_path, text="path:  ")
entry_path = tk.Entry(master=root, width=40)
frame_class = tk.Frame(master=root)

label_class = tk.Label(master=frame_class, text="class:  ")
frame_result = tk.Frame(master=root)
label_result = tk.Label(master=frame_result, text=" ")
button_transfer = tk.Button(master=root, text="classify!!", height=2, width=10)


frame_path.grid(row=0, column=0, padx=10)
label_path.grid(row=0, column=0)
entry_path.grid(row=0, column=1, padx=10)
frame_class.grid(row=2, column=0, padx=10)
label_class.grid(row=2, column=0)
frame_result.grid(row=2, column=1)
label_result.grid(row=2, column=1)
button_transfer.grid(row=1, column=1)


root.mainloop()