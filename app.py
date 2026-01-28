import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import gdown

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Image Authenticity Checker",
    layout="centered"
)

st.title("🛡️ AI Image Authenticity Checker")

# ================= LIMITATIONS (BIG & CLEAR) =================
st.markdown(
    """
    🚨 **IMPORTANT LIMITATIONS – PLEASE READ** 🚨  

    **This app only detects faces that are created or modified using AI.**  
    It does NOT analyze backgrounds, objects, or non-face regions.

    **This app may NOT correctly identify real images captured using iPhone cameras or images with strong filters or enhancements.**  
    Results for such images should be interpreted with caution.
    """
)

st.write(
    "**Hello, please upload your image to test if this is real or AI generated/modified.**"
)

st.caption(
    "Images are processed temporarily and are not stored. "
    "Results are probabilistic and for awareness purposes only."
)

# ================= SETTINGS =================
THRESHOLD = 0.65
device = torch.device("cpu")

MODEL_URL = "https://drive.google.com/uc?id=1aYVR0bisFExbdX7ZPYpjXN6TL5K6Fsb9"
MODEL_PATH = "model_2_gemini.pth"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model (first-time setup)...")
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
uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# ================= PREDICTION =================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=320)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = 1 - fake_prob

    st.subheader("🔍 Result")

    if fake_prob >= THRESHOLD:
        st.error(
            f"❌ **Image seems fake**\n\n"
            f"**Confidence:** {fake_prob * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ **Image appears real**\n\n"
            f"**Confidence:** {real_prob * 100:.2f}%"
        )

    st.caption(
        "⚠️ This app does not identify the specific AI tool used. "
        "It only estimates whether a face in the image is AI-generated or real."
    )
