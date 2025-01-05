from curses import panel
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
from PIL import Image, ImageTk
import cv2 as ocv
import numpy as np
import cv2
import matplotlib.pyplot as plt

# data set
img_cv_list = []  
img_pil_list = [] 
image_refs = []  
images_reset = [] 
THUMBNAIL_SIZE = (100, 100)

# normal image
img_cv = None
img_pil = None
img_original_cv = None

# method about dataset
# ---------------------------------------------------------------------------------------------------------start of method---------------------------------------------------------------------------------------------------#
def upload_folder():
    global img_cv_list, img_pil_list, image_refs, images_reset
    folder_path = filedialog.askdirectory()
    if folder_path:
        img_cv_list = []  
        img_pil_list = [] 
        image_refs = []  
        
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img = Image.open(file_path)#pillow image
                img.thumbnail(THUMBNAIL_SIZE)  
                img_pil_list.append(img) 
                img_cv_list.append(ocv.imread(file_path)) 
                images_reset = img_cv_list.copy()
                
            
        display_images_dataset()  
def reset_images_dataset():
    global img_cv_list, img_pil_list, images_reset

    if not images_reset:
        print("No images to reset. Please load images first.")
        return

    img_cv_list = images_reset.copy()
    img_pil_list = []

    for img_cv in img_cv_list:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))  
        img_pil.thumbnail(THUMBNAIL_SIZE)  
        img_pil_list.append(img_pil)

    display_images_dataset()

def lines_dataset():
    global img_cv_list, img_pil_list
    if not img_cv_list:
        print("No images loaded. Please load images first.")
        return

    for idx, img_cv in enumerate(img_cv_list):
        img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
        h, w = img_new.shape
        img_cv = img_new.copy()
        
        for row in range(h):
            for col in range(w):
                if row == col or (row + col) == 300:  
                    img_new[row, col] = 255  

        img_cv_list[idx] = ocv.cvtColor(img_new, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    display_images_dataset() 
def display_images_dataset(event=None):
    global img_pil_list, image_refs
    canvas.delete("all")  
    
    x_offset = 10
    y_offset = 10
    canvas_width = canvas.winfo_width()
    
    image_refs = []  
    
    for img_pil in img_pil_list:
        img_tk = ImageTk.PhotoImage(img_pil)
        image_refs.append(img_tk)  
        
        if x_offset + THUMBNAIL_SIZE[0] > canvas_width: 
            x_offset = 10
            y_offset += THUMBNAIL_SIZE[1] + 10
        
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
        x_offset += THUMBNAIL_SIZE[0] + 10
    
    canvas.config(scrollregion=(0, 0, canvas_width, y_offset + THUMBNAIL_SIZE[1]))

def apply_negative_filter_dataset():
    global img_cv_list, img_pil_list
    if not img_cv_list:
        print("No images loaded. Please load images first.")
        return

    for idx, img_cv in enumerate(img_cv_list):
        img_new_dataset_dataset = ocv.cvtColor(img_cv, ocv.COLOR_BGR2GRAY)
        img_new_dataset_dataset = 255 - img_new_dataset_dataset  
        
        img_cv_list[idx] = ocv.cvtColor(img_new_dataset_dataset, ocv.COLOR_GRAY2BGR)  
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])  

        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)

    display_images_dataset()

def power_low_dataset():
    global img_cv_list, img_pil_list
    if not img_cv_list:
        print("No images loaded. Please load images first.")
        return

    for idx, img_cv in enumerate(img_cv_list):
        img_new_dataset_dataset = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
        img_new_dataset_dataset = 3 * (img_new_dataset_dataset ** 0.8)
        img_new_dataset_dataset = np.clip(img_new_dataset_dataset, 0, 255).astype(np.uint8)
        
        img_cv_list[idx] = ocv.cvtColor(img_new_dataset_dataset, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
   
    display_images_dataset()


def log_transformation_dataset():
    global img_cv_list, img_pil_list
    if not img_cv_list:
        print("No image loaded. Please load an image first.")
        return
    for idx, img_cv in enumerate(img_cv_list):
        img_new_dataset_dataset = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
        img_new_dataset_dataset = 30 * np.log2(img_new_dataset_dataset + 1)
        img_new_dataset_dataset = np.clip(img_new_dataset_dataset, 0, 255).astype(np.uint8)
        
        img_cv_list[idx] = ocv.cvtColor(img_new_dataset_dataset, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)

    display_images_dataset()
   
def save_images_dataset():
    """Save all processed images to a selected folder."""
    global img_pil_list
    if not img_pil_list:
        print("No images to save. Please upload images first.")
        return

    save_folder = filedialog.askdirectory()
    if not save_folder:
        return  

    for idx, img_pil in enumerate(img_pil_list):
        save_path = os.path.join(save_folder, f"image_{idx + 1}.png")
        img_pil.save(save_path)  
        print(f"Saved: {save_path}")  

def histogram_dataset():
    global img_cv_list, img_pil_list
    if not img_cv_list:
        print("No image loaded. Please load an image first.")
        return
    for idx, img_cv in enumerate(img_cv_list):
        img_new_dataset_dataset = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
        l=256
        counts=np.bincount(img_new_dataset_dataset.flatten(),minlength=l)
        pdf= counts/np.size(img_new_dataset_dataset)

        cdf=np.cumsum(pdf)
        norm =cdf*(l-1)
        norm=np.round(norm)
        norm=norm.astype(np.uint8)

        new_mat = norm[img_new_dataset_dataset]
        
        img_cv_list[idx] = ocv.cvtColor(new_mat, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    display_images_dataset()
    
def apply_filter_dataset(filter_type):
    global img_cv_list, img_pil_list
    if not img_pil_list:
        print("No images to save. Please upload images first.")
        return
    for idx, img_cv in enumerate(img_cv_list):
        img_new_dataset = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
        h, w = img_new_dataset.shape

        filter_size = 3
        offset = filter_size // 2
        img_filtered = img_new_dataset.copy().astype(np.float32)
        
        for row in range(offset, h - offset):
            for col in range(offset, w - offset):
                area = img_new_dataset[row - offset:row + offset + 1, col - offset:col + offset + 1]
                
                if filter_type == "average":
                    img_filtered[row, col] = np.average(area)
                elif filter_type == "maximum":
                    img_filtered[row, col] = np.max(area)
                elif filter_type == "minimum":
                    img_filtered[row, col] = np.min(area)
                elif filter_type == "median":
                    img_filtered[row, col] = np.median(area)
        
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
        img_cv_list[idx] = ocv.cvtColor(img_filtered, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
        img_new_dataset = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
   
    display_images_dataset()



def app_Increase_Decrease_dataset(filter_type_):
    global img_cv_list, img_pil_list
    if img_pil_list is None:
        print("No image loaded. Please load an image first.")
        return

    for idx, img_cv in enumerate(img_cv_list):
        img_new_dataset = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
        
    
        img_new_float_dataset = img_new_dataset.astype(np.float32)
        
        if filter_type_ == "increase":
            img_new_float_dataset *= 2  
        elif filter_type_ == "decrease":
            img_new_float_dataset /= 2 
        
        img_new_float_dataset = np.clip(img_new_float_dataset, 0, 255).astype(np.uint8)
        
        img_cv_list[idx] = ocv.cvtColor(img_new_float_dataset, ocv.COLOR_GRAY2RGB)
        
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    
    display_images_dataset()
def apply_sobel_filter_dataset():
    global img_cv_list, img_pil_list
    if img_pil_list is None:
        print("No image loaded. Please load an image first.")
        return
    for idx, img_cv in enumerate(img_cv_list):
        img_gray = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
        h, w = img_gray.shape

        Gx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

        Gy = np.array([[-1, -2, -1],
                    [0,   0,  0],
                    [1,   2,  1]], dtype=np.float32)

        filteredImg1 = np.zeros_like(img_gray)
        filteredImg2 = np.zeros_like(img_gray)

        filter_size = 3
        offset = filter_size // 2

        for row in range(offset, h - offset):
            for col in range(offset, w - offset):
                area = img_gray[row - offset: row + offset + 1, col - offset: col + offset + 1]
                filteredImg1[row, col] = np.sum(area * Gx) ** 2
                filteredImg2[row, col] = np.sum(area * Gy) ** 2

        filteredImg3 = np.sqrt(filteredImg1 + filteredImg2)

        if np.max(filteredImg3) != 0:
            filteredImg3 = (filteredImg3 / np.max(filteredImg3)) * 255
        filteredImg3 = np.clip(filteredImg3, 0, 255).astype(np.uint8)

        img_cv_list[idx] = ocv.cvtColor(filteredImg3, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    display_images_dataset()

def apply_prewitt_filter_datset():
    global img_cv_list, img_pil_list
    if img_cv_list is None:
        print("No image loaded. Please load an image first.")
        return
    for idx, img_cv in enumerate(img_cv_list):
        img_gray = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
        h, w = img_gray.shape

        Gx = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]], dtype=np.float32)

        Gy = np.array([[-1, -1, -1],
                    [0,   0,  0],
                    [1,   1,  1]], dtype=np.float32)

        filteredImg1 = np.zeros_like(img_gray)
        filteredImg2 = np.zeros_like(img_gray)

        filter_size = 3
        offset = filter_size // 2

        for row in range(offset, h - offset):
            for col in range(offset, w - offset):
                area = img_gray[row - offset: row + offset + 1, col - offset: col + offset + 1]
                filteredImg1[row, col] = np.sum(area * Gx) ** 2
                filteredImg2[row, col] = np.sum(area * Gy) ** 2

        filteredImg3 = np.sqrt(filteredImg1 + filteredImg2)

        if np.max(filteredImg3) != 0:
            filteredImg3 = (filteredImg3 / np.max(filteredImg3)) * 255
        filteredImg3 = np.clip(filteredImg3, 0, 255).astype(np.uint8)

        img_cv_list[idx] = ocv.cvtColor(filteredImg3, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])    
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    display_images_dataset()



def apply_Gaussian_filter_dataset():

    global img_cv_list, img_pil_list
    if img_cv_list is None:
        print("No image loaded. Please load an image first.")
        return
    for idx, img_cv in enumerate(img_cv_list):
        img_gray = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
        h, w = img_gray.shape

        
        gx=np.array([[1/16,2/16,1/6],
        [2/16,4/16,2/16],
        [1/16,2/16,1/16]], dtype=np.float32)

        new = np.zeros_like(img_gray)
        

        filter_size = 3
        offset = filter_size // 2

        for row in range(offset, h - offset):
            for col in range(offset, w - offset):
                area = img_gray[row - offset: row + offset + 1, col - offset: col + offset + 1]
                new[row, col] = np.sum(area * gx)
    
        

        if np.max(new) != 0:
            new = (new / np.max(new)) * 255
        new = np.clip(new, 0, 255).astype(np.uint8)

        img_cv_list[idx] = ocv.cvtColor(new, ocv.COLOR_GRAY2RGB)
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    display_images_dataset()



def histo_matching_dataset():
    global img_cv_list, img_pil_list
    if img_cv_list is None:
        print("No image loaded. Please load an image first.")
        return
    
    for idx, img_cv in enumerate(img_cv_list):
        img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)

        Nr = np.bincount(img_new.flatten(), minlength=256) 
        Pdf_original = Nr / np.size(img_new)  
        Cdf_original = np.cumsum(Pdf_original) 


        Pz_target = np.array([0.10, 0.15, 0.15, 0.20, 0.20, 0.10, 0.05, 0.05])
        Cdf_target = np.cumsum(Pz_target)  


        L = 256  
        G_zq = np.round((L - 1) * Cdf_target).astype(np.uint8)


        mapping = np.zeros(256, dtype=np.uint8)  
        j = 0  

        for i in range(256):
            while j < len(G_zq) and Cdf_original[i] > Cdf_target[j]:
                j += 1
            mapping[i] = G_zq[j] if j < len(G_zq) else G_zq[-1]

        Result = mapping[img_new]
        
        img_cv_list[idx] = ocv.cvtColor(Result, ocv.COLOR_GRAY2RGB) 
        img_pil_list[idx] = Image.fromarray(img_cv_list[idx])  
        
        img_pil_list[idx] = img_pil_list[idx].resize(THUMBNAIL_SIZE)
    display_images_dataset()



# ---------------------------------------------------------------------------------------------------------- End of method of dataset -----------------------------------------------------------------------------------------------#
# method of normal image




def load_image():
    global img_cv, img_pil, img_original_cv
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        img_cv = ocv.imread(file_path)
        img_cv = ocv.resize(img_cv, (300, 300))
        img_cv = ocv.cvtColor(img_cv, ocv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)  
        img_original_cv = img_cv.copy()
        display_image(ImageTk.PhotoImage(img_pil))

def display_image(image):
    panel.config(image=image)
    panel.image = image



def apply_negative_filter():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
        
    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
    img_new = 255 - img_new  
    
    img_cv = ocv.cvtColor(img_new, ocv.COLOR_GRAY2RGB)  
    img_pil = Image.fromarray(img_cv) 
    display_image(ImageTk.PhotoImage(img_pil))
def apply_reset_image():
    global img_cv, img_pil, img_original_cv
    if img_original_cv is None:
        print("No image loaded. Please load an image first.")
        return
    
    img_cv = img_original_cv.copy()
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))

def log_transformation():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return

    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
    img_new = 30 * np.log2(img_new + 1)
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)
    
    img_cv = ocv.cvtColor(img_new, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))

def power_low():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
        
    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
    img_new = 3 * (img_new ** 0.8)
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)
    
    img_cv = ocv.cvtColor(img_new, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))



def apply_filter(filter_type):
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
        
    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
    h, w = img_new.shape

    filter_size = 3
    offset = filter_size // 2
    img_filtered = img_new.copy().astype(np.float32)
    
    for row in range(offset, h - offset):
        for col in range(offset, w - offset):
            area = img_new[row - offset:row + offset + 1, col - offset:col + offset + 1]
            
            if filter_type == "average":
                img_filtered[row, col] = np.average(area)
            elif filter_type == "maximum":
                img_filtered[row, col] = np.max(area)
            elif filter_type == "minimum":
                img_filtered[row, col] = np.min(area)
            elif filter_type == "median":
                img_filtered[row, col] = np.median(area)
    
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    img_cv = ocv.cvtColor(img_filtered, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))

def save_image():
    global img_pil
    if img_pil is None:
        print("No image to save. Please load or modify an image first.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            filetypes=[("PNG files", ".png"), ("JPEG files", ".jpg")])
    if file_path:
        img_pil.save(file_path)  

def app_Increase_Decrease(filter_type_):
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return

    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
    
   
    img_new_float = img_new.astype(np.float32)
    
    if filter_type_ == "increase":
        img_new_float *= 2  
    elif filter_type_ == "decrease":
        img_new_float /= 2 
    
    img_new_float = np.clip(img_new_float, 0, 255).astype(np.uint8)
    
    img_cv = ocv.cvtColor(img_new_float, ocv.COLOR_GRAY2RGB)
    
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))



def apply_sobel_filter():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
    
    img_gray = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
    h, w = img_gray.shape

    Gx = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[-1, -2, -1],
                  [0,   0,  0],
                  [1,   2,  1]], dtype=np.float32)

    filteredImg1 = np.zeros_like(img_gray)
    filteredImg2 = np.zeros_like(img_gray)

    filter_size = 3
    offset = filter_size // 2

    for row in range(offset, h - offset):
        for col in range(offset, w - offset):
            area = img_gray[row - offset: row + offset + 1, col - offset: col + offset + 1]
            filteredImg1[row, col] = np.sum(area * Gx) ** 2
            filteredImg2[row, col] = np.sum(area * Gy) ** 2

    filteredImg3 = np.sqrt(filteredImg1 + filteredImg2)

    if np.max(filteredImg3) != 0:
        filteredImg3 = (filteredImg3 / np.max(filteredImg3)) * 255
    filteredImg3 = np.clip(filteredImg3, 0, 255).astype(np.uint8)

    img_cv = ocv.cvtColor(filteredImg3, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))
    


def apply_prewitt_filter():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
    
    img_gray = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
    h, w = img_gray.shape

    Gx = np.array([[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[-1, -1, -1],
                  [0,   0,  0],
                  [1,   1,  1]], dtype=np.float32)

    filteredImg1 = np.zeros_like(img_gray)
    filteredImg2 = np.zeros_like(img_gray)

    filter_size = 3
    offset = filter_size // 2

    for row in range(offset, h - offset):
        for col in range(offset, w - offset):
            area = img_gray[row - offset: row + offset + 1, col - offset: col + offset + 1]
            filteredImg1[row, col] = np.sum(area * Gx) ** 2
            filteredImg2[row, col] = np.sum(area * Gy) ** 2

    filteredImg3 = np.sqrt(filteredImg1 + filteredImg2)

    if np.max(filteredImg3) != 0:
        filteredImg3 = (filteredImg3 / np.max(filteredImg3)) * 255
    filteredImg3 = np.clip(filteredImg3, 0, 255).astype(np.uint8)

    img_cv = ocv.cvtColor(filteredImg3, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))
    
def apply_Gaussian_filter():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
    
    img_gray = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY).astype(np.float32)
    h, w = img_gray.shape

    
    gx=np.array([[1/16,2/16,1/6],
     [2/16,4/16,2/16],
     [1/16,2/16,1/16]], dtype=np.float32)

    new = np.zeros_like(img_gray)
    

    filter_size = 3
    offset = filter_size // 2

    for row in range(offset, h - offset):
        for col in range(offset, w - offset):
            area = img_gray[row - offset: row + offset + 1, col - offset: col + offset + 1]
            new[row, col] = np.sum(area * gx)

    if np.max(new) != 0:
        new = (new / np.max(new)) * 255
    new = np.clip(new, 0, 255).astype(np.uint8)

    img_cv = ocv.cvtColor(new, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))

def histo_matching():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
    
    Nr = np.bincount(img_new.flatten(), minlength=256) 
    Pdf_original = Nr / np.size(img_new)  
    Cdf_original = np.cumsum(Pdf_original) 


    Pz_target = np.array([0.10, 0.15, 0.15, 0.20, 0.20, 0.10, 0.05, 0.05])
    Cdf_target = np.cumsum(Pz_target)  


    L = 256  
    G_zq = np.round((L - 1) * Cdf_target).astype(np.uint8)


    mapping = np.zeros(256, dtype=np.uint8)  
    j = 0  

    for i in range(256):
        while j < len(G_zq) and Cdf_original[i] > Cdf_target[j]:
            j += 1
        mapping[i] = G_zq[j] if j < len(G_zq) else G_zq[-1]

    Result = mapping[img_new]
    
    img_cv = ocv.cvtColor(Result, ocv.COLOR_GRAY2RGB) 
    img_pil = Image.fromarray(img_cv)  
    display_image(ImageTk.PhotoImage(img_pil))



def histogram():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
    l=256
    counts=np.bincount(img_new.flatten(),minlength=l)
    pdf= counts/np.size(img_new)

    cdf=np.cumsum(pdf)
    norm =cdf*(l-1)
    norm=np.round(norm)
    norm=norm.astype(np.uint8)

    new_mat = norm[img_new]
    
    img_cv = ocv.cvtColor(new_mat, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))





def lines():
    global img_cv, img_pil
    if img_cv is None:
        print("No image loaded. Please load an image first.")
        return
    
    img_new = ocv.cvtColor(img_cv, ocv.COLOR_RGB2GRAY)
    h, w = img_new.shape
    img_cv = img_new.copy()
    
    for row in range(h):
        for col in range(w):
            if row == col or (row + col) == 300:  
                img_new[row, col] = 255  

    img_cv = ocv.cvtColor(img_new, ocv.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_cv)
    display_image(ImageTk.PhotoImage(img_pil))  

#  window of normal image  -------------------------------------------------------------------------------------------------------------------------------




root2 = tk.Tk()
root2.title("Image Filters GUI")
root2.geometry("1000x600")

frame_top = tk.Frame(root2, bg="lightblue", height=50)
frame_top.pack(side=tk.TOP, fill=tk.X)

btn_load = tk.Button(frame_top, text="Load Image", command=load_image, width=20)
btn_load.pack(pady=10)

frame_left_buttons = tk.Frame(root2, bg="lightgrey", width=150)
frame_left_buttons.pack(side=tk.LEFT, fill=tk.Y)

frame_image = tk.Frame(root2, bg="white")
frame_image.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_right_buttons = tk.Frame(root2, bg="lightgrey", width=150)
frame_right_buttons.pack(side=tk.RIGHT, fill=tk.Y)

# image Display Panel
panel = tk.Label(frame_image, bg="white")
panel.pack(pady=20)

# Control Buttons - Left Side
btn_negative = tk.Button(frame_left_buttons, text="Negative Filter", command=apply_negative_filter, width=15)
btn_negative.pack(pady=10)

btn_power_low = tk.Button(frame_left_buttons, text="Power Low Filter", command=power_low, width=15)
btn_power_low.pack(pady=10)

btn_log_transformation = tk.Button(frame_left_buttons, text="Log Transformation", command=log_transformation, width=15)
btn_log_transformation.pack(pady=10)

btn_Gaussian = tk.Button(frame_left_buttons, text="Gaussian Filter", command=apply_Gaussian_filter, width=15)
btn_Gaussian.pack(pady=10)

btn_histo_matching = tk.Button(frame_left_buttons, text="Histogram Matching", command=histo_matching, width=15)
btn_histo_matching.pack(pady=10)

btn_histogram = tk.Button(frame_left_buttons, text="Histogram", command=histogram, width=15)
btn_histogram.pack(pady=10)

btn_prewitt = tk.Button(frame_left_buttons, text="prewitt Filter", command=apply_prewitt_filter, width=15)
btn_prewitt.pack(pady=10)

btn_lines = tk.Button(frame_left_buttons, text="Adding X line", command= lines, width=15)
btn_lines.pack(pady=10)

# Control Buttons - Right Side
btn_average = tk.Button(frame_right_buttons, text="Average Filter", command=lambda: apply_filter("average"), width=15)
btn_average.pack(pady=10)

btn_maximum = tk.Button(frame_right_buttons, text="Maximum Filter", command=lambda: apply_filter("maximum"), width=15)
btn_maximum.pack(pady=10)

btn_minimum = tk.Button(frame_right_buttons, text="Minimum Filter", command=lambda: apply_filter("minimum"), width=15)
btn_minimum.pack(pady=10)

btn_median = tk.Button(frame_right_buttons, text="Median Filter", command=lambda: apply_filter("median"), width=15)
btn_median.pack(pady=10)


btn_sobel = tk.Button(frame_right_buttons, text="Sobel Filter", command=apply_sobel_filter, width=15)
btn_sobel.pack(pady=10)


btn_increase = tk.Button(frame_right_buttons, text="Increase", command=lambda: app_Increase_Decrease("increase"), width=15)
btn_increase.pack(pady=10)

btn_decrease = tk.Button(frame_right_buttons, text="Decrease", command=lambda: app_Increase_Decrease("decrease"), width=15)
btn_decrease.pack(pady=10)

btn_reset = tk.Button(frame_right_buttons, text="Reset Image", command=apply_reset_image, width=15)
btn_reset.pack(pady=10)

btn_save = tk.Button(frame_top, text="Save Image", command=save_image, width=15)
btn_save.pack(pady=10)


root2.mainloop()





# Create the main window
# root = tk.Tk()
# root.title("Upload Folder and Display Images")

# # Create a frame for the canvas and scrollbar
# frame = Frame(root)
# frame.pack(fill=BOTH, expand=True)

# # Create the canvas to display images
# canvas = tk.Canvas(frame, bg="white")
# canvas.pack(side=LEFT, fill=BOTH, expand=True)

# # Add a vertical scrollbar to the canvas
# scl = Scrollbar(frame, orient="vertical", command=canvas.yview)
# scl.pack(side=RIGHT, fill=Y)

# canvas.config(yscrollcommand=scl.set)

# # Bind the window resize event to adjust the layout dynamically
# canvas.bind("<Configure>", display_images_dataset)

# # Buttons

# upload_button_dataset = tk.Button(root, text="Upload Folder", command=upload_folder, bg="black", fg="white")
# upload_button_dataset.pack(pady=10)

# btn_histogram_dataset = tk.Button(root, text="Histogram", command=histogram_dataset, width=15)
# btn_histogram_dataset.pack(pady=10)

# btn_negative_dataset = tk.Button(root, text="Negative Filter", command=apply_negative_filter_dataset, bg="black", fg="white")
# btn_negative_dataset.pack(side=LEFT, padx=20, pady=10)

# btn_power_low_dataset = tk.Button(root, text="Power Low Filter", command=power_low_dataset, width=15)
# btn_power_low_dataset.pack(pady=10)

# btn_log_transformation_dataset = tk.Button(root, text="Log Transformation", command=log_transformation_dataset, width=15)
# btn_log_transformation_dataset.pack(pady=10)

# btn_save_dataset = tk.Button(root, text="Save Images", command=save_images_dataset, bg="blue", fg="white")
# btn_save_dataset.pack(pady=10)

# # text="Negative Filter", command=apply_negative_filter_dataset, bg="green", fg="white", font=("Arial", 12), height=2, width=20
# btn_average_dataset = tk.Button(root, text="Average Filter", command=lambda: apply_filter_dataset("average"), width=15)
# btn_average_dataset.pack(pady=10)

# btn_maximum_dataset = tk.Button(root, text="Maximum Filter", command=lambda: apply_filter_dataset("maximum"), width=15)
# btn_maximum_dataset.pack(pady=10)

# btn_maximum_dataset = tk.Button(root, text="Minimum Filter", command=lambda: apply_filter_dataset("minimum"), width=15)
# btn_maximum_dataset.pack(pady=10)

# btn_median_dataset = tk.Button(root, text="Median Filter", command=lambda: apply_filter_dataset("median"), width=15)
# btn_median_dataset.pack(pady=10)

# btn_increase_dataset = tk.Button(root, text="Increase", command=lambda: app_Increase_Decrease_dataset("increase"), width=15)
# btn_increase_dataset.pack(pady=10)

# btn_decrease_dataset = tk.Button(root, text="Decrease", command=lambda: app_Increase_Decrease_dataset("decrease"), width=15)
# btn_decrease_dataset.pack(pady=10)


# btn_sobel_dataset = tk.Button(root, text="Sobel Filter", command=apply_sobel_filter_dataset, bg="black", fg="white")
# btn_sobel_dataset.pack(padx=10)

# btn_prewitt_dataset = tk.Button(root, text="prewitt Filter", command=apply_prewitt_filter_datset,bg="black", fg="white", width=15)
# btn_prewitt_dataset.pack(pady=0)

# root.mainloop()


root = tk.Tk()
root.title("Upload Folder and Display Images")

# canvas and scrollbar
frame = Frame(root)
frame.pack(fill=BOTH, expand=True)

#  canvas to display images
canvas = tk.Canvas(frame, bg="white")
canvas.pack(side=LEFT, fill=BOTH, expand=True)

#  vertical scrollbar to the canvas
scl = Scrollbar(frame, orient="vertical", command=canvas.yview)
scl.pack(side=RIGHT, fill=Y)

canvas.config(yscrollcommand=scl.set)

canvas.bind("<Configure>", display_images_dataset)

# frame hold all buttons
button_frame = Frame(root, bg="lightgray")
button_frame.pack(fill=BOTH, expand=False, pady=10)


# method dynamically arrange buttons in rows and columns
def arrange_buttons_in_grid(buttons, frame, max_columns=5):
    for i, button in enumerate(buttons):
        row = i // max_columns
        col = i % max_columns
        button.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

# our buttons
upload_button_dataset = tk.Button(button_frame, text="Upload Folder", command=upload_folder, bg="black", fg="white")
btn_negative_dataset = tk.Button(button_frame, text="Negative Filter", command=apply_negative_filter_dataset, bg="black", fg="white")
btn_power_low_dataset = tk.Button(button_frame, text="Power Low Filter", command=power_low_dataset, width=15)
btn_log_transformation_dataset = tk.Button(button_frame, text="Log Transformation", command=log_transformation_dataset, width=15)
btn_save_dataset = tk.Button(button_frame, text="Save Images", command=save_images_dataset, bg="blue", fg="white")
btn_average_dataset = tk.Button(button_frame, text="Average Filter", command=lambda: apply_filter_dataset("average"), width=15)
btn_maximum_dataset = tk.Button(button_frame, text="Maximum Filter", command=lambda: apply_filter_dataset("maximum"), width=15)
btn_minimum_dataset = tk.Button(button_frame, text="Minimum Filter", command=lambda: apply_filter_dataset("minimum"), width=15)
btn_median_dataset = tk.Button(button_frame, text="Median Filter", command=lambda: apply_filter_dataset("median"), width=15)
btn_increase_dataset = tk.Button(button_frame, text="Increase", command=lambda: app_Increase_Decrease_dataset("increase"), width=15)
btn_decrease_dataset = tk.Button(button_frame, text="Decrease", command=lambda: app_Increase_Decrease_dataset("decrease"), width=15)
btn_sobel_dataset = tk.Button(button_frame, text="Sobel Filter", command=apply_sobel_filter_dataset, bg="black", fg="white")
btn_prewitt_dataset = tk.Button(button_frame, text="Prewitt Filter", command=apply_prewitt_filter_datset, bg="black", fg="white", width=15)
btn_Gaussian_dataset= tk.Button(button_frame, text="Gaussian Filter", command=apply_Gaussian_filter_dataset, bg="black", fg="white", width=15)
btn_histo_dataset = tk.Button(button_frame, text="Histogram" , command=histogram_dataset, bg="black", fg="white", width=15)
btn_histo_matching_dataset = tk.Button(button_frame, text="Histogram Matching", command=histo_matching_dataset, bg="black", fg="white", width=15)
btn_reset_dataset = tk.Button(button_frame, text="Reset Images", command=reset_images_dataset, bg="black", fg="white", width=15)
btn_line_dataset = tk.Button(button_frame, text="draw x line" , command=lines_dataset, bg="black", fg="white", width=15)


buttons = [
    upload_button_dataset, btn_negative_dataset, btn_power_low_dataset, btn_log_transformation_dataset, btn_save_dataset,
    btn_average_dataset, btn_maximum_dataset, btn_minimum_dataset, btn_median_dataset, btn_increase_dataset, btn_decrease_dataset, btn_sobel_dataset,
    btn_prewitt_dataset, btn_Gaussian_dataset, btn_histo_dataset, btn_histo_matching_dataset, btn_reset_dataset, btn_line_dataset
]

arrange_buttons_in_grid(buttons, button_frame, max_columns=5)
root.mainloop()
# ----------------------------------------------------------------------------- End of data set ----------------------------------------------------------------#

#  done :)  ------------------------------------- 






















