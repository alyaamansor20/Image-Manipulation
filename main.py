import random
import statistics
from tkinter.ttk import Style

import PIL
import math
import matplotlib.pyplot as plt
import numpy as np

from tkinter import *
from scipy import fftpack
from PIL import Image, ImageDraw, ImageTk


# ------------------------------------------------------------------------------------------------------
# --------- K-Means ---------
def get_centroids(clusters_number, image_size):
    centroid = []
    random.seed(7)
    for i in range(clusters_number):
        centroid.append(random.randrange(0, image_size))

    print("Centroids: ", centroid)

    return centroid


def get_clusters(centroid, image_data, error):
    cluster_point = [[] for i in range(len(centroid))]
    cluster_value = [[] for i in range(len(centroid))]

    for i in range(len(image_data)):
        # find distance between the pixel and every centroid
        distance = []
        for c in centroid:
            dist = 0
            for color in range(3):
                dist += math.pow((image_data[c][color] - image_data[i][color]), 2)
            dist = math.sqrt(dist)
            distance.append(dist)
        # find cluster that the point belongs to
        min_dist = min(distance)
        index = distance.index(min_dist)
        cluster_point[index].append(i)
        cluster_value[index].append(min_dist)

    for cluster in cluster_value:
        max_dist = max(cluster)
        if max_dist > error:
            return cluster_point, 1

    return cluster_point, 0


def k_means(image, clusters_number, error, epochs):
    # get flattened image consists of tuples (RGB)
    image_data = list(image.getdata())

    centroid = get_centroids(clusters_number, len(image_data))

    epoch_count = 0
    cluster = []
    cont = 1
    while (epoch_count != epochs) and cont:
        cluster, cont = get_clusters(centroid, image_data, error)

        centroid.clear()
        for c in cluster:
            x_points = []
            y_points = []
            for point in c:
                y = point // image.width
                x = point - (y * image.width)
                x_points.append(x)
                y_points.append(y)

            mean_x = int(statistics.mean(x_points))
            mean_y = int(statistics.mean(y_points))
            centroid.append((mean_y * image.width) + mean_x)

        epoch_count += 1

        print("Epoch #", epoch_count, centroid)

    for index in range(len(cluster)):
        for point in cluster[index]:
            image_data[point] = image_data[centroid[index]]

    image.putdata(image_data)

    return image


# ------------------------------------ Convolution START ---------------------------------------------------
def ApplyFilter(img, kernel, size):
    offset = len(kernel) // 2

    if img.mode == "L":
        # applay any filter to Gray scale image
        for x in range(offset, img.width - offset):
            for y in range(offset, img.height - offset):
                acc = 0
                for a in range(size):
                    for b in range(size):
                        xn = x + a - offset
                        yn = y + b - offset
                        pixel = img.getpixel((xn, yn))
                        acc += pixel * kernel[a][b]
                img.putpixel((x, y), int(acc))
        return img

    else :
        out_img = Image.new(mode="RGB", size=img.size)
        draw = ImageDraw.Draw(out_img)
        pixels = img.load()
        # applay any filter to RGB image
        for x in range(offset, img.width - offset):
            for y in range(offset, img.height - offset):
                acc = [0, 0, 0]
                for a in range(size):
                    for b in range(size):
                        xn = x + a - offset
                        yn = y + b - offset
                        pixel = pixels[xn, yn]
                        acc[0] += pixel[0] * kernel[a][b]
                        acc[1] += pixel[1] * kernel[a][b]
                        acc[2] += pixel[2] * kernel[a][b]
                draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
        return out_img
# ------------------------------------ Convolution END ---------------------------------------------------


# --------------------------------------- Brightness & Darkness START -----------------------------------------------
def apply_value_RGB(image, value):
    List = [0, 0, 0]
    pixels = image.load()
    draw = ImageDraw.Draw(image)
    for y in range(0, image.width):
        for x in range(0, image.height):
            pixel = pixels[y, x]
            List[0] = pixel[0] * value
            List[1] = pixel[1] * value
            List[2] = pixel[2] * value
            draw.point((y, x), (int(List[0]), int(List[1]), int(List[2])))
    return image


def apply_value_gray_scale(image, value):
    List = 0
    draw = ImageDraw.Draw(image)
    for y in range(0, image.width):
        for x in range(0, image.height):
            pixel = image.getpixel((y, x))
            List = pixel * value
            image.putpixel((y,x), int(List))
    return image


def enhance_image(img,value , mode):
    out_img = Image.new(img.mode, img.size)

    if img.mode == "L" :
        if mode == "brightness":
            out_img = apply_value_gray_scale(img, value)

        elif mode == "darkness":
            out_img = apply_value_gray_scale(img, (1 / value))

    else :
        if mode == "brightness":
            out_img = apply_value_RGB(img, value)

        elif mode == "darkness":
            out_img = apply_value_RGB(img, (1 / value))

    return out_img


# --------------------------------------- Brightness & Darkness END -----------------------------------------------

# -------------------------------------------- Band Reject Filter START----------------------------------------------


def CreateFilterImage(image_np):
    # Create a band reject filter image
    x, y = image_np.shape[1], image_np.shape[0]

    # size of circle
    outter_x, outter_y = 100, 100
    inner_x, inner_y = 40, 40

    # create a box
    outterBox = ((x / 2) - (outter_x / 2), (y / 2) - (outter_y / 2), (x / 2) + (outter_x / 2), (y / 2) + (outter_y / 2))
    innerBox = ((x / 2) - (inner_x / 2), (y / 2) - (inner_y / 2), (x / 2) + (inner_x / 2), (y / 2) + (inner_y / 2))

    band_reject = Image.new("L", (image_np.shape[1], image_np.shape[0]), color=1)

    draw1 = ImageDraw.Draw(band_reject)
    draw1.ellipse(outterBox, fill=0)
    draw1.ellipse(innerBox, fill=1)

    band_pass_np = np.array(band_reject)

    plt.imshow(band_reject)
    plt.show()

    return band_pass_np


def ApplyBandRejectFilter(image1):
    # convert image to numpy array
    image_np = np.array(image1)
    band_pass_np = CreateFilterImage(image_np)

    if image1.mode == "L":
        # apply fft
        fft = fftpack.fftshift(fftpack.fft2(image_np))
        # multiply both images (original image & filter image)
        filtered = np.multiply(fft, band_pass_np)
        ifft2 = fftpack.ifft2(fftpack.ifftshift(filtered))
        ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
        ifft2 = np.maximum(0, np.minimum(ifft2, 255))

        return ifft2.astype(np.uint8)

    else:
        r, g, b = image1.split()
        # convert R , G , B to numpy array
        red_np = np.array(r)
        green_np = np.array(g)
        blue_np = np.array(b)
        # apply fft on R , G , B
        fftRed = fftpack.fftshift(fftpack.fft2(red_np))
        fftGreen = fftpack.fftshift(fftpack.fft2(green_np))
        fftBlue = fftpack.fftshift(fftpack.fft2(blue_np))
        # multiply both the images
        filteredRed = np.multiply(fftRed, band_pass_np)
        filteredGreen = np.multiply(fftGreen, band_pass_np)
        filteredBlue = np.multiply(fftBlue, band_pass_np)
        # -------------------------------------------------
        ifft2R = fftpack.ifft2(fftpack.ifftshift(filteredRed))
        ifft2G = fftpack.ifft2(fftpack.ifftshift(filteredGreen))
        ifft2B = fftpack.ifft2(fftpack.ifftshift(filteredBlue))
        # -------------------------------------------------
        ifft2R = np.real(fftpack.ifft2(fftpack.ifftshift(filteredRed)))
        ifft2G = np.real(fftpack.ifft2(fftpack.ifftshift(filteredGreen)))
        ifft2B = np.real(fftpack.ifft2(fftpack.ifftshift(filteredBlue)))
        # -------------------------------------------------
        ifft2R = np.maximum(0, np.minimum(ifft2R, 255))
        ifft2G = np.maximum(0, np.minimum(ifft2G, 255))
        ifft2B = np.maximum(0, np.minimum(ifft2B, 255))
        # -------------------------------------------------
        ifft2R = Image.fromarray(ifft2R).convert("L")
        ifft2G = Image.fromarray(ifft2G).convert("L")
        ifft2B = Image.fromarray(ifft2B).convert("L")
        # -------------------------------------------------
        newImg = Image.merge("RGB", (ifft2R, ifft2G, ifft2B))

        return newImg


# -------------------------------------------- Band Reject Filter END----------------------------------------------

# ---------------------------------------------- Histogram Equalization START-------------------------------------


def plot_single_channel_image_histogram(single_channel_image, max_intensity_level=256, super_title='Histogram', title=''):
    plt.suptitle(super_title)
    plt.subplot(1, 1, 1)
    plt.hist(single_channel_image.getdata(), max_intensity_level, [0, max_intensity_level - 1])
    plt.title(title)
    plt.show()


def plot_rgb_image_histogram(rgb_image, max_intensity_level=256, super_title='RGB Histogram'):
    single_channel_image_r, single_channel_image_g, single_channel_image_b = rgb_image.split()

    plot_single_channel_image_histogram(single_channel_image_r, max_intensity_level, super_title, 'Red')
    plot_single_channel_image_histogram(single_channel_image_g, max_intensity_level, super_title, 'Green')
    plot_single_channel_image_histogram(single_channel_image_b, max_intensity_level, super_title, 'Blue')


def Histogram_Equalization(single_channel_image, max_intensity_level=256):
    intensities = [0] * 256
    total_number_of_pixels = single_channel_image.width * single_channel_image.height

    # find intensities occurrences
    for x in range(single_channel_image.width):
        for y in range(single_channel_image.height):
            intensities[single_channel_image.getpixel((x, y))] += 1

    # find intensities cumulative
    for i in range(1, len(intensities)):
        intensities[i] += intensities[i - 1]

    # find intensities new values
    for i in range(0, len(intensities)):
        intensities[i] = intensities[i] * (max_intensity_level - 1)
        intensities[i] //= total_number_of_pixels

    for x in range(0, single_channel_image.width):
        for y in range(0, single_channel_image.height):
            single_channel_image.putpixel((x, y), intensities[single_channel_image.getpixel((x, y))])

    return single_channel_image


# ---------------------------------------------- Histogram Equalization END -------------------------------------

# ---------------------------------------- Create Windows START -------------------------------------------------

def createMainWindow():
    # Creating master Tkinter window
    mainWindow = Tk()
    mainWindow.geometry("500x500")

    style = Style(mainWindow)
    style.configure("Button", background="light green",
                    foreground="red", font=("arial", 10, "bold"))

    # Dictionary to create multiple buttons
    values = [("Image Segmentation using K-means", createKmeansWindow),
              ("Band Reject Filter", createBandRejectFilterWindow),
              ("Contrast using Histogram Equalization", createContrastWindow),
              ("Apply Filter using Convolution", createApplyFilterWindow),
              ("Brightness & Darkness", createBrightnessDarknessWindow),
              ("Display Histogram", createDisplayHistogramWindow)]

    for (text, command) in values:
        Button(mainWindow, text=text, command=lambda command=command: (mainWindow.destroy(), command())).pack(fill=X, side=TOP, pady=10, padx=20)

    mainloop()


def createApplyFilterWindow():
    applyFilterWindow = Tk()
    applyFilterWindow.geometry("500x300")

    kernel_size = Text(applyFilterWindow, height=1, width=30)
    kernel = Text(applyFilterWindow, height=3, width=30)
    T_image_path = Text(applyFilterWindow, height=1, width=30)
    l1 = Label(text="Enter kernel size (3*3 kernel with size 3): ")
    l3 = Label(text="Enter the 2D array kernel: ")
    l1.pack()
    kernel_size.pack()
    l3.pack()
    kernel.pack()
    l2 = Label(text="Enter Image path: ")
    l2.pack()
    T_image_path.pack()
    def Take_input():
        value1 = kernel_size.get("1.0", "end-1c")
        value2 = kernel.get("1.0", "end-1c")
        a=[int(j) for j in value2.split(" ")]

        a = np.array(a).reshape(int(value1), int(value1)).tolist()

        print(a)
        imgPath = T_image_path.get("1.0", "end-1c")
        newImage = Image.open(imgPath)
        resultImage = ApplyFilter(newImage,a,int(value1))

        resultImage.save("Apply_Filter_output.png")
        resultImage.thumbnail((300,300))

        canvas = Canvas(applyFilterWindow, width=resultImage.width, height=resultImage.height)
        canvas.pack()

        imgtk = ImageTk.PhotoImage(resultImage)
        canvas.create_image(20, 20, anchor=NW, image=imgtk)

    enter_btn = Button(applyFilterWindow, height=2,
                       width=20,
                       text="Enter",
                       command=lambda: Take_input())

    def Go_Back():
        applyFilterWindow.destroy()
        createMainWindow()

    back_btn = Button(applyFilterWindow, height=2,
                      width=20,
                      text="back",
                      command=lambda: Go_Back())

    enter_btn.pack()
    back_btn.pack()


def createBrightnessDarknessWindow():
    brightnessDarknessWindow = Tk()
    brightnessDarknessWindow.geometry("500x300")

    T_mode = Text(brightnessDarknessWindow, height=1, width=30)
    T_value= Text(brightnessDarknessWindow, height=1, width=30)
    T_image_path = Text(brightnessDarknessWindow, height=1, width=30)
    l1 = Label(text="Enter the mode (Brightness/Darkness): ")
    l3 = Label(text="Enter the value: ")
    l1.pack()
    T_mode.pack()
    l3.pack()
    T_value.pack()
    l2 = Label(text="Enter Image path: ")
    l2.pack()
    T_image_path.pack()
    def Take_input():
        T_mode_value = T_mode.get("1.0", "end-1c")
        value = T_value.get("1.0", "end-1c")
        imgPath = T_image_path.get("1.0", "end-1c")
        image=Image.open(imgPath)
        showImage = enhance_image(image,int(value),T_mode_value)
        showImage.save("Brightness or darkness output.png")
        showImage.thumbnail((300,300))
        canvas = Canvas(brightnessDarknessWindow, width=showImage.width, height=showImage.height)
        canvas.pack()

        imgtk = ImageTk.PhotoImage(showImage)
        canvas.create_image(20, 20, anchor=NW, image=imgtk)

    enter_btn = Button(brightnessDarknessWindow, height=2,
                       width=20,
                       text="Enter",
                       command=lambda: Take_input())

    def Go_Back():
        brightnessDarknessWindow.destroy()
        createMainWindow()

    back_btn = Button(brightnessDarknessWindow, height=2,
                      width=20,
                      text="back",
                      command=lambda: Go_Back())

    enter_btn.pack()
    back_btn.pack()


def createKmeansWindow():
    kmeansWindow = Tk()
    kmeansWindow.geometry("600x500")
    T_k = Text(kmeansWindow, height=1, width=10)
    T_image_path = Text(kmeansWindow, height=1, width=30)
    T_epoch = Text(kmeansWindow, height=1, width=10)
    T_error = Text(kmeansWindow, height=1, width=10)
    l1 = Label(text="Enter K : ")
    l2 = Label(text="Enter number of epochs: ")
    l3 = Label(text="Enter accepted error: ")
    l4 = Label(text="Enter Image path: ")


    def Take_input():
        k = T_k.get("1.0", "end-1c")
        epochs = T_epoch.get("1.0", "end-1c")
        error = T_error.get("1.0", "end-1c")
        imgPath = T_image_path.get("1.0", "end-1c")
        img = Image.open(imgPath)
        kmeans_img = k_means(img,int(k),int(error),int(epochs))

        kmeans_img.save("k-means_output.png")
        kmeans_img.thumbnail((250, 250))

        canvas = Canvas(kmeansWindow, width=img.width, height=img.height)
        canvas.pack()

        if img.mode == "L":
            imgtk = ImageTk.PhotoImage(image=PIL.Image.fromarray(kmeans_img))
        else:
            imgtk = ImageTk.PhotoImage(kmeans_img)
        canvas.create_image(20, 20, anchor=NW, image=imgtk)

    enter_btn = Button(kmeansWindow, height=2,
                    width=20,
                    text="Enter",
                    command=lambda: Take_input())


    def Go_Back():
        kmeansWindow.destroy()
        createMainWindow()

    back_btn = Button(kmeansWindow, height=2,
                       width=20,
                       text="back",
                       command=lambda: Go_Back())

    l1.pack()
    T_k.pack()
    l2.pack()
    T_epoch.pack()
    l3.pack()
    T_error.pack()
    l4.pack()
    T_image_path.pack()
    enter_btn.pack()
    back_btn.pack()


def createBandRejectFilterWindow():
    BandRejectFilterWindow = Tk()
    BandRejectFilterWindow.geometry("600x500")
    T_image_path = Text(BandRejectFilterWindow, height=1, width=30)
    label = Label(text="Enter Image path: ")

    def Take_input():
        imgPath = T_image_path.get("1.0", "end-1c")
        img = Image.open(imgPath)
        bandReject_img = ApplyBandRejectFilter(img)

        bandReject_img.save("band-reject_output.png")
        bandReject_img.thumbnail((250,250))

        canvas = Canvas(BandRejectFilterWindow,width = img.width, height = img.height)
        canvas.pack()

        print(bandReject_img.mode)
        if img.mode == "L":
            imgtk = ImageTk.PhotoImage(image=PIL.Image.fromarray(bandReject_img))
        else :
            imgtk = ImageTk.PhotoImage(bandReject_img)

        canvas.create_image(20, 20, anchor=NW, image=imgtk)

    enter_btn = Button(BandRejectFilterWindow, height=2,
                    width=20,
                    text="Enter",
                    command=lambda: Take_input())

    def Go_Back():
        BandRejectFilterWindow.destroy()
        createMainWindow()

    back_btn = Button(BandRejectFilterWindow, height=2,
                      width=20,
                      text="back",
                      command=lambda: Go_Back())

    label.pack()
    T_image_path.pack()
    enter_btn.pack()
    back_btn.pack()


def createContrastWindow():
    ContrastWindow = Tk()
    ContrastWindow.geometry("600x500")
    T_image_path = Text(ContrastWindow, height=1, width=30)
    label = Label(text="Enter Image path: ")

    def Take_input():
        imgPath = T_image_path.get("1.0", "end-1c")
        img = Image.open(imgPath)
        contrast_img = Histogram_Equalization(img.convert("L"))
        contrast_img.save("contrast_output.png")
        contrast_img.thumbnail((250,250))
        canvas = Canvas(ContrastWindow, width=img.width, height=img.height)
        canvas.pack()
        if img.mode == "L":
            imgtk = ImageTk.PhotoImage(image=PIL.Image.fromarray(contrast_img))
        else :
            imgtk = ImageTk.PhotoImage(contrast_img)

        canvas.create_image(20, 20, anchor=NW, image=imgtk)

    enter_btn = Button(ContrastWindow, height=2,
                    width=20,
                    text="Enter",
                    command=lambda: Take_input())

    def Go_Back():
        ContrastWindow.destroy()
        createMainWindow()

    back_btn = Button(ContrastWindow, height=2,
                      width=20,
                      text="back",
                      command=lambda: Go_Back())

    label.pack()
    T_image_path.pack()
    enter_btn.pack()
    back_btn.pack()


def createDisplayHistogramWindow():
    DisplayHistogramWindow = Tk()
    DisplayHistogramWindow.geometry("600x500")
    T_image_path = Text(DisplayHistogramWindow, height=1, width=30)
    label = Label(text="Enter Image path: ")

    def Take_input():
        imgPath = T_image_path.get("1.0", "end-1c")
        img = Image.open(imgPath)

        if img.mode == "L":
            plot_single_channel_image_histogram(img)
        else:
            plot_rgb_image_histogram(img)

    enter_btn = Button(DisplayHistogramWindow, height=2,
                    width=20,
                    text="Enter",
                    command=lambda: Take_input())

    def Go_Back():
        DisplayHistogramWindow.destroy()
        createMainWindow()

    back_btn = Button(DisplayHistogramWindow, height=2,
                      width=20,
                      text="back",
                      command=lambda: Go_Back())

    label.pack()
    T_image_path.pack()
    enter_btn.pack()
    back_btn.pack()


# ---------------------------------------- Create Windows END -------------------------------------------------

# ---------------------------------------- main -------------------------------------------------

createMainWindow()