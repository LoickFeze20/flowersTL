import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="CottonGuard Pro", page_icon="🌿", layout="wide")

# ============================================
# STYLE
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .header {
        background: linear-gradient(135deg, #0B2B26, #1a4a3a);
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .header h1 { font-size: 3rem; margin: 0; }
    .header p { opacity: 0.9; }
    
    .result-box {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .result-healthy { background: linear-gradient(135deg, #2e7d32, #1b5e20); }
    .result-disease { background: linear-gradient(135deg, #c62828, #8e0000); }
    .result-class { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
    .result-confidence { font-size: 3.5rem; font-weight: 800; margin: 0.5rem 0; }
    .result-date { font-size: 0.9rem; opacity: 0.8; }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
    .metric-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    .metric-value { font-size: 1.3rem; font-weight: 600; }
    .metric-label { font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; }
    
    .stButton > button {
        background: #0B2B26;
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(11,43,38,0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DONNÉES
# ============================================
CLASSES = [
    "Alternaria Leaf Spot",
    "Bacterial Blight", 
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt"
]

DISEASE_INFO = {
    "Alternaria Leaf Spot": {
        "severity": "Élevée",
        "type": "Fongique",
        "impact": "30% pertes"
    },
    "Bacterial Blight": {
        "severity": "Très élevée",
        "type": "Bactérienne",
        "impact": "50% pertes"
    },
    "Fusarium Wilt": {
        "severity": "Élevée",
        "type": "Fongique",
        "impact": "40% pertes"
    },
    "Healthy Leaf": {
        "severity": "Aucune",
        "type": "Sain",
        "impact": "Optimal"
    },
    "Verticillium Wilt": {
        "severity": "Élevée",
        "type": "Fongique",
        "impact": "35% pertes"
    }
}

# ============================================
# TON MODÈLE AVEC ATTENTION
# ============================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=[2, 3])
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SNAttentionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = CBAM(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = CBAM(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attention3 = CBAM(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.attention4 = CBAM(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.attention3(x)
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.attention4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load('sn_attention.pth', map_location='cpu')
        model = SNAttentionModel(num_classes=5)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {str(e)}")
        return None

def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ============================================
# INTERFACE
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🌿 CottonGuard Pro</h1>
        <p>Diagnostic des maladies du coton par IA • Transfert Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    # Main
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Image")
        uploaded = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, caption="Image chargée", use_container_width=True)
            
            if st.button("🚀 Analyser", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    model = load_model()
                    
                    if model:
                        input_tensor = preprocess_image(image)
                        
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs[0], dim=0)
                            pred_idx = torch.argmax(probs).item()
                            confidence = probs[pred_idx].item() * 100
                        
                        st.session_state.pred_idx = pred_idx
                        st.session_state.confidence = confidence
                        st.session_state.probs = probs.numpy()
                        st.session_state.date = datetime.now().strftime("%d/%m/%Y • %H:%M")
                        st.rerun()
    
    with col2:
        if 'pred_idx' in st.session_state:
            pred = CLASSES[st.session_state.pred_idx]
            info = DISEASE_INFO[pred]
            display_date = st.session_state.get('date', datetime.now().strftime("%d/%m/%Y • %H:%M"))
            
            box_class = "result-healthy" if "Healthy" in pred else "result-disease"
            st.markdown(f"""
            <div class="result-box {box_class}">
                <div style="opacity:0.8;">DIAGNOSTIC</div>
                <div class="result-class">{pred}</div>
                <div class="result-confidence">{st.session_state.confidence:.1f}%</div>
                <div class="result-date">{display_date}</div>
                <div class="metric-grid">
                    <div class="metric-item"><div class="metric-value">{info['severity']}</div><div class="metric-label">Sévérité</div></div>
                    <div class="metric-item"><div class="metric-value">{info['type']}</div><div class="metric-label">Type</div></div>
                    <div class="metric-item"><div class="metric-value">{info['impact']}</div><div class="metric-label">Impact</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Graphique
            fig = go.Figure()
            colors = ['#c62828', '#c62828', '#c62828', '#2e7d32', '#c62828']
            
            for i, (cls, prob) in enumerate(zip(CLASSES, st.session_state.probs)):
                fig.add_trace(go.Bar(
                    x=[cls[:15] + "..." if len(cls) > 15 else cls],
                    y=[prob * 100],
                    marker_color=colors[i],
                    text=[f"{prob*100:.1f}%"],
                    textposition='outside',
                    showlegend=False
                ))
            
            fig.update_layout(
                height=300,
                title="Probabilités par classe",
                yaxis_title="%",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌿 CottonGuard Pro • Transfert Learning </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()