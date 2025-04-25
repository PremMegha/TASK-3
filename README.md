# TASK-3 NEURAL STYLE TRANSFER 

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PREM SUMERUKUMAR MEGHA

*INTERN ID*: C0DF232

*DOMAIN*: Artificial Intelligence Markup Language.

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH

---

## Overview

During my four-week AI internship at Codtech IT Solutions, I designed and built a Neural Style Transfer (NST) application in Python for TASK-3. NST is a technique that merges the content of one image (your photo) with the style of another image (like a painting) to produce a creative blend. My goal was to make this powerful method accessible through an easy-to-use graphical interface.

## What I Used and Learned:-

To complete this project, I:-
- Followed **YouTube** tutorials and **GeeksforGeeks** articles to understand how NST works under the hood.  
- Asked **ChatGPT** for guidance on Tkinter GUI design and debugging tips.  
- Used **Tkinter** to build a simple window with buttons to load images and apply the style transfer.  
- Leveraged **PyTorch** with a pretrained VGG-19 model to extract content and style features.  
- Add **OpenCV** to play a looping video in the background for a modern look.

Along the way, I:-
1. Learned how to load, resize, and normalize images.  
2. Extracted feature maps from specific VGG-19 layers.  
3. Computed content and style losses using mean squared error and Gram matrices.  
4. Optimized an initial image with the Adam optimizer to combine style and content.  
5. Ensured the GUI remains responsive by running the style transfer on a separate thread.

## Requirements

- **OS**: Windows 10 or above
- **Python**: 3.13.3 (required)  
- **Optional**: CUDA and a GPU (e.g., NVIDIA GTX 1650) for faster processing

## How to Run:-

 - Click **Select Content** and choose your photo.  
 - Click **Select Style** and choose an artwork or image for style.  
 - Click **Apply Style** to start processing.  
 - Watch the status bar to see progress percentage.  
 - When it finishes, the stylized image will appear and save as `output.png`.

## Project Structure

```
neural-style-transfer/
├── styles/           # predefined Style images stored in this
├── NST.py            # Main script: GUI and style transfer code
├── requirements.txt  # Libraries needed
└── README.md         # Project documentation
```

- **styles/** contains sample images with different artistic looks (anime, comic, ghibli etc.).  
- **NST.py** handles image loading, model setup, optimization loop, and GUI.


## OUTPUT:-

![GUI]
![Image](https://github.com/user-attachments/assets/6f7e8cd1-5863-45a4-b82a-4d92c0b1fd86)


![pop_art style]
![Image](https://github.com/user-attachments/assets/d847b743-e82e-4ac9-a96d-b55bcc8a5e35)

![ghibli style]
![Image](https://github.com/user-attachments/assets/cde9a18b-d51e-4ba5-9a56-90b2fabc5a46)

![comic style]
![Image](https://github.com/user-attachments/assets/b2b4f2e8-7e76-41ff-8234-efc81278c2f3)

![painting style]
![Image](https://github.com/user-attachments/assets/99d5df31-adf6-4a1b-9c6d-84da1b087f3c)


## Real-World Uses

- **Digital Art**: Artists can experiment by blending photos with painting styles.  
- **Social Media**: Create unique visuals for marketing or personal posts.  
- **Education**: Demonstrate deep learning concepts in a hands-on way.
