# C:\Users\hp\handwritten-digit-cgan\pytorch-MNIST-CelebA-cGAN-cDCGAN\app\streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# ==================== PROFESSIONAL PAGE CONFIG ====================
st.set_page_config(
    page_title="Handwritten Digit Generator | AI 4009",
    page_icon="pen",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Handwritten Digit Generator using cGAN & cDCGAN\nAI 4009 – Fall 2025"
    }
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {font-size: 3.5rem !important; color: #1E88E5; text-align: center; font-weight: bold;
    .subtitle {font-size: 1.4rem; color: #546E7A; text-align: center; margin-bottom: 2rem;}
    .stButton>button {background: linear-gradient(90deg, #1E88E5, #42A5F5); color: white; font-weight: bold; border: none; height: 3em; border-radius: 12px;}
    .stButton>button:hover {background: linear-gradient(90deg, #1565C0, #1E88E5);}
    .css-1d391kg {padding-top: 2rem;}
    .generated-box {border: 3px solid #42A5F5; border-radius: 15px; padding: 15px; background-color: #f8fbff;}
    .footer {text-align: center; margin-top: 4rem; color: #78909C;}
</style>
""", unsafe_allow_html=True)

# ==================== TITLE & HEADER ====================
st.markdown('<h1 class="main-header">Handwritten Digit Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Conditional GAN & Conditional DCGAN trained on MNIST Dataset</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== YOUR EXACT GENERATOR CLASSES ====================
def normal_init(m, mean, std):
    if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

class CGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, noise, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(noise)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)

class cDCGAN_Generator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2,1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, noise, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(noise)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x

# ==================== LOAD MODELS ====================
@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cgan = CGAN_Generator().to(device)
    cdcgan = cDCGAN_Generator(d=128).to(device)
    
    cgan_path = "models/generator_param.pkl"                    # ← removed "../"
    cdcgan_path = "models/MNIST_cDCGAN_generator_param.pkl"

    if not os.path.exists(cgan_path):
        st.error(f"cGAN model not found at {cgan_path}")
    if not os.path.exists(cdcgan_path):
        st.error(f"cDCGAN model not found at {cdcgan_path}")
    cgan.load_state_dict(torch.load(cgan_path, map_location=device))
    cdcgan.load_state_dict(torch.load(cdcgan_path, map_location=device))
    
    cgan.eval()
    cdcgan.eval()
    
    return {"cgan": cgan, "cdcgan": cdcgan}, device

with st.spinner("Loading trained models..."):
    models, device = load_models()
st.success("Models loaded successfully!")

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("Control Panel")
    st.markdown("### Model Selection")
    model_choice = st.radio("Choose Architecture", ["cDCGAN (Best Quality)", "cGAN (Faster)"], index=0)
    model_key = "cdcgan" if "cDCGAN" in model_choice else "cgan"
    
    st.markdown("### Input")
    number = st.text_input("Enter Number (0–9)", value="2025", max_chars=15, help="Up to 15 digits")
    
    st.markdown("### Style")
    n_vars = st.slider("Number of Variations", 1, 8, 4)
    
    st.markdown("---")
    generate_btn = st.button("Generate Handwritten Number", type="primary", use_container_width=True)

# ==================== GENERATION FUNCTIONS ====================
def generate_single_digit(model, digit, is_cdcgan):
    model.eval()
    with torch.no_grad():
        if is_cdcgan:
            noise = torch.randn(1, 100, 1, 1, device=device)
            label = torch.zeros(1, 10, 1, 1, device=device)
            label[0, digit] = 1.0
        else:
            noise = torch.randn(1, 100, device=device)
            label = torch.zeros(1, 10, device=device)
            label[0, digit] = 1.0
            
        img_tensor = model(noise, label)
        img_tensor = (img_tensor + 1) / 2
        img = img_tensor[0, 0].cpu().numpy()
    return (img * 255).astype(np.uint8)

def combine_digits(digits_imgs):
    imgs = [Image.fromarray(d) for d in digits_imgs]
    w, h = 28, 28
    total_w = len(imgs) * w + (len(imgs)-1) * 25
    combined = Image.new("L", (total_w, h), 255)
    x = 0
    for img in imgs:
        combined.paste(img, (x, 0))
        x += w + 25
    return combined

# ==================== MAIN GENERATION ====================
if generate_btn:
    if not number.isdigit():
        st.error("Please enter digits only (0–9)!")
    else:
        digits = [int(d) for d in number]
        model = models[model_key]
        is_cdcgan = (model_key == "cdcgan")

        with st.spinner(f"Creating magic with {model_choice}..."):
            results = []
            for i in range(n_vars):
                digit_imgs = [generate_single_digit(model, d, is_cdcgan) for d in digits]
                final_img = combine_digits(digit_imgs)
                results.append(final_img)

        st.markdown("---")
        st.markdown(f"<h2 style='text-align:center;color:#1E88E5;'>Your Handwritten Number: <b>{number}</b></h2>", unsafe_allow_html=True)
        
        cols = st.columns(min(n_vars, 4))
        for idx, img in enumerate(results):
            with cols[idx % 4]:
                st.image(img, use_container_width=True)
                st.caption(f"Variation {idx+1}")

        # Download
        buf = io.BytesIO()
        results[0].save(buf, format="PNG")
        st.download_button(
            label="Download Image as PNG",
            data=buf.getvalue(),
            file_name=f"handwritten_{number}_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png",
            use_container_width=True
        )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h4>AI 4009 – Generative Artificial Intelligence | Fall 2025</h4>
</div>
""", unsafe_allow_html=True)

st.balloons()
