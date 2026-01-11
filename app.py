import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import gdown

# ================= PAGE SETUP =================
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🕵️ Deepfake Image Detector")
st.caption("For research and educational purposes only")

# ================= SETTINGS =================
THRESHOLD = 0.65
device = torch.device("cpu")

# ================= MODEL DOWNLOAD =================
MODEL_URL = "https://drive.google.com/uc?id=1aYVR0bisFExbdX7ZPYpjXN6TL5K6Fsb9"
MODEL_PATH = "model_2_gemini.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model (first time only)...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = timm.create_model("xception", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ================= IMAGE UPLOAD =================
uploaded = st.file_uploader(
    "Upload an image to check if it is REAL or FAKE",
    type=["jpg", "jpeg", "png"]
)

# ================= PREDICTION =================
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    fake_prob = probs[0][0].item()  # 0 = FAKE

    st.subheader("Result")

    if fake_prob >= THRESHOLD:
        st.error(f"🚨 FAKE IMAGE\n\nConfidence: {fake_prob*100:.2f}%")
    else:
        st.success(f"✅ REAL IMAGE\n\nConfidence: {(1 - fake_prob)*100:.2f}%")
