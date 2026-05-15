
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models.hybrid_model import HybridDetector
import tempfile

# Page config
st.set_page_config(
    page_title="DeepGuard",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background-color: #050816;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #00E5FF;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #B0BEC5;
    margin-bottom: 40px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.real {
    background-color: rgba(0,255,150,0.15);
    border: 1px solid #00FF95;
    color: #00FF95;
}

.fake {
    background-color: rgba(255,0,80,0.15);
    border: 1px solid #FF0055;
    color: #FF4D6D;
}

.confidence {
    text-align: center;
    font-size: 18px;
    color: #E0E0E0;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🛡️ DeepGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Deepfake Detection System</div>', unsafe_allow_html=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = HybridDetector().to(device)

model.load_state_dict(
    torch.load("deepguard_model.pth", map_location=device)
)

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Classes
classes = ['ai_generated', 'real']


# Upload section
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:

        tmp_file.write(uploaded_file.getbuffer())

        temp_path = tmp_file.name

    # Open safely
    image = Image.open(temp_path).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    # Transform
    img = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing image..."):

        with torch.no_grad():
            output = model(img)

        pred = torch.argmax(output, 1).item()

        confidence = torch.softmax(output, dim=1)[0][pred].item()

    prediction = classes[pred]

    if prediction == "real":

        st.markdown(
            '''
            <div class="result-box real">
            ✅ REAL IMAGE
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '''
            <div class="result-box fake">
            🚨 AI GENERATED IMAGE
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="confidence">Confidence: {confidence:.2%}</div>',
        unsafe_allow_html=True
    )

    st.write(uploaded_file.type)

    # Result UI
    if prediction == "real":

        st.markdown(
            '''
            <div class="result-box real">
            ✅ REAL IMAGE
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '''
            <div class="result-box fake">
            🚨 AI GENERATED IMAGE
            </div>
            ''',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="confidence">Confidence: {confidence:.2%}</div>',
        unsafe_allow_html=True
    )

    # Result UI
    if prediction == "real":

        st.markdown(
            f'''<div class="result-box real">
            ✅ REAL IMAGE
            </div>''',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'''<div class="result-box fake">
            🚨 AI GENERATED IMAGE
            </div>''',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="confidence">Confidence: {confidence:.2%}</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.caption("DeepGuard • AI Deepfake Detection Project")