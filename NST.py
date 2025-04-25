import tkinter as tk
from tkinter import filedialog, ttk, Label, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import threading
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
import os
import cv2
import numpy as np

# Device configuration (force GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size configuration
image_size = 768  # Further reduced to fit GPU memory

# Define transform for input images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])

# Function to load an image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

# Function to convert tensor back to image
def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Load pretrained VGG19 model for style transfer
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:21].to(device).eval()

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if layer_num in {0, 5, 10, 19}:  # relu1_1, relu2_1, relu3_1, relu4_1
                features.append(x)
        return features

# Function to calculate Gram matrix
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# This function performs neural style transfer by blending content and style features using VGG19.
# It returns the path to the generated output image.
def style_transfer(content_path, style_path, progress_callback=None, output_path='output.png', num_steps=400, style_weight=5e4, content_weight=1):
    content = load_image(content_path)
    style = load_image(style_path)

    model = VGGFeatures()

    target = content.clone().to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=0.01)  # Lower learning rate for finer tuning

    content_features = model(content)
    style_features = model(style)
    style_grams = [gram_matrix(f) for f in style_features]

    print("Starting style transfer...")
    for step in range(num_steps):
        target_features = model(target)
        content_loss = torch.mean((target_features[-1] - content_features[-1])**2)
        style_loss = 0

        for t_feat, s_gram in zip(target_features, style_grams):
            t_gram = gram_matrix(t_feat)
            style_loss += torch.mean((t_gram - s_gram)**2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        print(f"Step {step}/{num_steps}, Loss: {total_loss.item():.2f}")

        if progress_callback and step % 10 == 0:
            progress_callback(step, num_steps)

    output = target.clone().detach().cpu().clamp_(0, 1)
    save_image(output, output_path)
    return output_path

# -------- GUI CODE --------

class StyleTransferApp:
    def __init__(self, master):
        self.master = master
        master.title("Neural Style Transfer - NST.py")
        master.geometry("1000x700")

        # Set a colorful geometric background image like Huemint theme
        import cv2
        import numpy as np
        self.cap = cv2.VideoCapture(r"C:\AI-ML\TASK3\Blue and White Modern Technology Innovation Presentation.mp4")
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (1000, 700))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.bg_photo = ImageTk.PhotoImage(image)
        self.bg_label = tk.Label(master, image=self.bg_photo)
        self.update_video_background()
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Color palette
        primary_bg = "#cdd3d2"
        primary_fg = "#1b1b1a"
        highlight = "#eb4509"
        accent = "#3897d1"

        self.content_path = None
        self.style_path = None

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TButton",
                        font=("Segoe UI", 12),
                        padding=10,
                        background=primary_fg,
                        foreground="white")

        self.container = tk.Frame(master, bg=primary_bg, highlightbackground=accent, highlightthickness=4)
        self.container_bg_label = tk.Label(self.container, image=self.bg_photo)
        self.container_bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.container.place(relx=0.5, rely=0.5, anchor="center", width=700, height=550)

        self.content_label = Label(self.container, text="Upload Content Image:", bg=primary_bg, fg=primary_fg, font=("Segoe UI", 14, "bold"))
        self.content_label.pack(pady=5)
        self.content_button = ttk.Button(self.container, text="Select Content", command=self.load_content_image)
        self.content_button.pack(pady=5)

        self.style_label = Label(self.container, text="Select Style Image:", bg=primary_bg, fg=primary_fg, font=("Segoe UI", 14, "bold"))
        self.style_label.pack(pady=5)
        self.style_button = ttk.Button(self.container, text="Select Style", command=self.load_style_image)
        self.style_button.pack(pady=5)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.style_options = {
            "Anime": os.path.join(base_dir, "styles", "anime.jpg"),
            "Comic": os.path.join(base_dir, "styles", "comic.jpeg"),
            "Ghibli": os.path.join(base_dir, "styles", "ghibli.jpg"),
            "Marvel": os.path.join(base_dir, "styles", "marvel.jpg"),
            "Paint": os.path.join(base_dir, "styles", "paint.jpg"),
            "Pop Art": os.path.join(base_dir, "styles", "pop_art.jpg"),
            "Starry Night": os.path.join(base_dir, "styles", "starry_night.jpeg")
        }

        self.style_choice = tk.StringVar()
        self.style_dropdown = ttk.Combobox(self.container, textvariable=self.style_choice, values=list(self.style_options.keys()), font=("Segoe UI", 12), height=len(self.style_options))
        self.style_dropdown.set("Choose Predefined Style (Optional)")
        self.style_dropdown.pack(pady=10, fill=tk.X)

        self.start_button = ttk.Button(self.container, text="Apply Style", command=self.apply_style)
        self.start_button.pack(pady=20)

        self.status_label = Label(self.container, text="", bg=primary_bg, fg=highlight, font=("Segoe UI", 12, "bold"))
        self.status_label.pack(pady=10)

    def load_content_image(self):
        self.content_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        self.status_label.config(text=f"Content Loaded: {os.path.basename(self.content_path)}")

    def load_style_image(self):
        self.style_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        self.status_label.config(text=f"Style Loaded: {os.path.basename(self.style_path)}")

    def update_progress(self, step, total_steps):
        percent = int((step / total_steps) * 100)
        self.status_label.config(text=f"Processing... {percent}% complete")
        self.master.update_idletasks()

    def update_video_background(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        frame = cv2.resize(frame, (1000, 700))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.bg_photo = ImageTk.PhotoImage(image)
        self.bg_label.configure(image=self.bg_photo)
        self.bg_label.image = self.bg_photo
        self.master.after(33, self.update_video_background)

    def apply_style(self):
        content = self.content_path
        style = self.style_path or self.style_options.get(self.style_choice.get())

        if not content or not style:
            self.status_label.config(text="Please load both content and style images!")
            return

        self.status_label.config(text="Processing... 0% complete")
        self.master.update_idletasks()

        threading.Thread(target=self.run_style_transfer, args=(content, style), daemon=True).start()

    # This method runs the style transfer in a separate thread to prevent the GUI from freezing during processing.
    def run_style_transfer(self, content, style):
        try:
            output_path = style_transfer(content, style, progress_callback=self.update_progress)
            self.status_label.config(text="Style applied! Output saved as output.png")
            output_img = Image.open(output_path)
            output_img.show()
            messagebox.showinfo("Done", "Style transfer completed and saved as output.png")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            messagebox.showerror("Error", str(e))

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()
