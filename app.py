# File: app.py
# Tujuan: Backend Flask untuk image denoising menggunakan HalfUNet model (.pth)
# Model: best_model.pth atau optimized_halfunet_physical.pth

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename

# ============================================================================
# MODEL ARCHITECTURE (HalfUNet - sama seperti di train.ipynb)
# ============================================================================

FILTER = 128

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class HalfUNet(nn.Module):
    def __init__(self, input_channels=3):
        super(HalfUNet, self).__init__()
        self.initial = nn.Conv2d(3, FILTER, 1, 1)
        self.conv1 = nn.Sequential(NAFBlock(FILTER), NAFBlock(FILTER))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(NAFBlock(FILTER), NAFBlock(FILTER))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(NAFBlock(FILTER), NAFBlock(FILTER))
        self.conv_up3 = nn.Conv2d(FILTER, FILTER * 16, 1, bias=False)
        self.up3 = nn.PixelShuffle(4)
        self.conv_up2 = nn.Conv2d(FILTER, FILTER * 4, 1, bias=False)
        self.up2 = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(FILTER, 3, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.conv1(x)
        pool1 = self.pool1(x1)
        x2 = self.conv2(pool1)
        pool2 = self.pool2(x2)
        x3 = self.conv3(pool2)
        up3 = self.conv_up3(x3)
        up3 = self.up3(up3)
        up2 = self.conv_up2(x2)
        up2 = self.up2(up2)
        up_scaled = x1 + up2 + up3
        output = self.final_conv(up_scaled)
        return output

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')
CORS(app)  # Enable CORS untuk akses dari frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variables
model = None
device = None
MODEL_PATH = 'optimized_halfunet_physical.pth'  # Default model path

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_from_file(model_path=MODEL_PATH):
    """
    Load HalfUNet model from .pth file
    
    Args:
        model_path: Path to model checkpoint (.pth file)
    
    Returns:
        model: Loaded PyTorch model
        device: Device (cuda/cpu)
    """
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"   Available models in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.pth'):
                print(f"   - {f}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Initialize model
    model = HalfUNet()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if checkpoint contains model_state_dict or is direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from checkpoint:")
            print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
            if 'val_psnr' in checkpoint:
                print(f"   - Val PSNR: {checkpoint['val_psnr']:.2f} dB")
            if 'val_ssim' in checkpoint:
                print(f"   - Val SSIM: {checkpoint['val_ssim']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Model loaded (direct state_dict)")
        
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Model file: {model_path}")
        print("=" * 70)
        
        return model, device
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size (height, width)
    
    Returns:
        tensor: Preprocessed image tensor [1, 3, H, W]
        original_size: Original image size (H, W)
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Store original size
    original_size = image.shape[:2]  # (H, W)
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, original_size

def postprocess_image(tensor, original_size=None):
    """
    Postprocess model output to image
    
    Args:
        tensor: Model output tensor [1, 3, H, W]
        original_size: Optional target size to resize back (H, W)
    
    Returns:
        image: Numpy array in [0, 255] range
    """
    # Convert tensor to numpy: (1, C, H, W) -> (H, W, C)
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Clip to [0, 1]
    image = np.clip(image, 0, 1)
    
    # Resize back to original size if specified
    if original_size is not None:
        image = cv2.resize(image, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
    
    # Convert to [0, 255]
    image = (image * 255).astype(np.uint8)
    
    return image

def sharpen_image(image, method='unsharp', strength=1.0):
    """
    Sharpen image to enhance details after denoising
    
    Args:
        image: Numpy array (H, W, C) in [0, 255] range
        method: Sharpening method ('unsharp', 'laplacian', 'adaptive')
        strength: Sharpening strength (0.5 - 2.0, default 1.0)
    
    Returns:
        sharpened: Sharpened image in [0, 255] range
    """
    # Normalize strength to reasonable range
    strength = np.clip(strength, 0.5, 2.0)
    
    if method == 'unsharp':
        # Unsharp masking - most natural looking
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Calculate sharpening mask
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        
    elif method == 'laplacian':
        # Laplacian sharpening - more aggressive
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
        
        # Add laplacian to original
        sharpened = img_float - strength * laplacian
        
    elif method == 'adaptive':
        # Adaptive sharpening - preserves smooth areas
        # Detect edges using Sobel
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize edge map to [0, 1]
        edge_map = edge_magnitude / (edge_magnitude.max() + 1e-8)
        edge_map = np.clip(edge_map, 0, 1)
        
        # Create adaptive strength map
        strength_map = edge_map * strength
        
        # Apply unsharp masking with adaptive strength
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        diff = image.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply adaptive sharpening
        sharpened = image.astype(np.float32)
        for c in range(3):  # For each color channel
            sharpened[:, :, c] += diff[:, :, c] * strength_map
    
    else:
        # Default: return original
        return image
    
    # Clip to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

def numpy_to_base64(image):
    """
    Convert numpy array to base64 string
    
    Args:
        image: Numpy array (H, W, C) in [0, 255] range
    
    Returns:
        base64_str: Base64 encoded string
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return base64_str

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'model_path': MODEL_PATH
    })

@app.route('/restore', methods=['POST'])
def restore():
    """
    Main restoration endpoint
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with original and restored images as base64
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
            }), 500
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Please upload an image.'
            }), 400
        
        file = request.files['image']
        
        # Check if file has filename
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected.'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        original_image = np.array(image.convert('RGB'))
        
        print(f"📥 Received image: {file.filename} | Size: {original_image.shape}")
        
        # Preprocess
        input_tensor, original_size = preprocess_image(original_image)
        input_tensor = input_tensor.to(device)
        
        # Get sharpening parameters from request (optional)
        sharpen_method = request.form.get('sharpen_method', 'unsharp')  # 'unsharp', 'laplacian', 'adaptive', 'none'
        sharpen_strength = float(request.form.get('sharpen_strength', 1.2))  # 0.5 - 2.0
        
        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess
        restored_image = postprocess_image(output_tensor, original_size)
        
        # Apply sharpening if requested
        if sharpen_method != 'none':
            print(f"🔧 Applying {sharpen_method} sharpening (strength: {sharpen_strength})")
            restored_image = sharpen_image(restored_image, method=sharpen_method, strength=sharpen_strength)
        
        # Convert to base64
        original_base64 = numpy_to_base64(original_image)
        restored_base64 = numpy_to_base64(restored_image)
        
        print(f"✓ Processing complete | Output size: {restored_image.shape}")
        
        # Return result
        return jsonify({
            'success': True,
            'original_image': f'data:image/png;base64,{original_base64}',
            'restored_image': f'data:image/png;base64,{restored_base64}',
            'original_size': original_size,
            'message': 'Image denoised successfully!'
        })
    
    except Exception as e:
        print(f"❌ Error during restoration: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Alternative upload endpoint that saves file and returns path"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'HalfUNet',
        'parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'model_path': MODEL_PATH,
        'input_size': [256, 256],
        'output_channels': 3
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Load model at startup
    print("\n🚀 Starting Flask Application...")
    
    try:
        # Check for available models and choose the best one
        available_models = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not available_models:
            print("❌ No .pth model files found in current directory!")
            print("   Please ensure 'best_model.pth' or 'optimized_halfunet_physical.pth' exists.")
            exit(1)
        
        # Priority: optimized_halfunet_physical.pth > best_model.pth > other .pth files
        if 'optimized_halfunet_physical.pth' in available_models:
            MODEL_PATH = 'optimized_halfunet_physical.pth'
            print("✓ Using optimized_halfunet_physical.pth (pre-trained model)")
        elif 'best_model.pth' in available_models:
            MODEL_PATH = 'best_model.pth'
            print("✓ Using best_model.pth (your trained model)")
        else:
            MODEL_PATH = available_models[0]
            print(f"✓ Using {MODEL_PATH}")
        
        model, device = load_model_from_file(MODEL_PATH)
        print("\n✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {str(e)}")
        print("   The application will start but /restore endpoint will not work.")
        model = None
        device = None
    
    print("\n" + "=" * 70)
    print("🌐 Flask server starting...")
    print("   - Main page: http://127.0.0.1:5000/")
    print("   - API endpoint: http://127.0.0.1:5000/restore")
    print("   - Health check: http://127.0.0.1:5000/health")
    print("=" * 70)
    print("\n💡 Press CTRL+C to stop the server\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
