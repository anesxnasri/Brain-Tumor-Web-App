import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠")

st.title("🧠 Brain Tumor Segmentation AI")
st.write("Upload an MRI image to detect the tumor area using SegFormer (MiT-B2).")

@st.cache_resource
def load_model():
    device = torch.device("cpu") 
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    try:
        model.load_state_dict(torch.load("SegFormer_MiTB2_Best.pth", map_location=device))
    except FileNotFoundError:
        st.error("Error: Model file 'best_model.pth' not found.")
        return None
        
    model.to(device)
    model.eval()
    return model

model = load_model()

IMAGE_SIZE = 352
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    if st.button("🔍 Detect Tumor"):
        with st.spinner("Analyzing..."):
            
            img_np = np.array(image)
            
            
            augmented = transform(image=img_np)["image"].unsqueeze(0)
            
            
            with torch.no_grad():
                output = model(augmented)
                pred_mask = torch.sigmoid(output).squeeze().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            
            
            img_resized = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))
            
            
            mask_colored = np.zeros_like(img_resized)
            mask_colored[:, :, 0] = pred_mask
            
            
            overlay = cv2.addWeighted(img_resized, 0.7, mask_colored, 0.3, 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(pred_mask, caption="Predicted Mask", use_column_width=True)
            with col2:
                st.image(overlay, caption="Tumor Detection (Red Area)", use_column_width=True)
            
            st.success("Analysis Completed! ✅")
