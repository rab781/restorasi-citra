# Image Denoising Web Application
## HalfUNet Deep Learning Model

Aplikasi web untuk menghilangkan noise dari gambar menggunakan model HalfUNet yang telah dilatih.

## 📋 Features

- ✨ Image denoising menggunakan deep learning
- 🖼️ Drag & drop interface
- 📊 Side-by-side comparison (original vs restored)
- 💾 Download hasil restoration
- 🚀 Fast inference dengan PyTorch
- 📱 Responsive design

## 🏗️ Arsitektur

**Model:** HalfUNet dengan NAFBlock
- Filter: 128 channels
- Architecture: Encoder-Decoder dengan PixelShuffle upsampling
- Input: RGB images (256×256)
- Output: Denoised RGB images

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan Model File Ada

Aplikasi akan mencari model file dalam urutan prioritas:
1. `best_model.pth` (hasil training Anda)
2. `optimized_halfunet_physical.pth` (model yang sudah ada)
3. File `.pth` lainnya

Pastikan salah satu file model ada di direktori yang sama dengan `app.py`.

## 🚀 Usage

### 1. Jalankan Flask Server

```bash
python app.py
```

Server akan berjalan di: `http://127.0.0.1:5000/`

### 2. Akses Web Interface

Buka browser dan navigasi ke: `http://127.0.0.1:5000/`

### 3. Upload & Process

1. Drag & drop gambar atau klik "Choose Image"
2. Klik "Denoise Image" untuk memproses
3. Lihat hasil comparison
4. Download gambar yang sudah di-restore

## 🔌 API Endpoints

### `GET /`
Serve main web interface

### `GET /health`
Health check endpoint
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda",
    "model_path": "best_model.pth"
}
```

### `POST /restore`
Main denoising endpoint

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
    "success": true,
    "original_image": "data:image/png;base64,...",
    "restored_image": "data:image/png;base64,...",
    "original_size": [height, width],
    "message": "Image denoised successfully!"
}
```

### `GET /model/info`
Get model information
```json
{
    "model_type": "HalfUNet",
    "parameters": 12345678,
    "device": "cuda",
    "model_path": "best_model.pth",
    "input_size": [256, 256],
    "output_channels": 3
}
```

## 📁 File Structure

```
d:\Raihan\
├── app.py                              # Flask backend
├── requirements.txt                    # Python dependencies
├── best_model.pth                      # Your trained model
├── optimized_halfunet_physical.pth     # Alternative model
├── templates/
│   └── index.html                      # Web interface
└── uploads/                            # Uploaded images (auto-created)
```

## ⚙️ Configuration

### Model Selection
Edit `MODEL_PATH` in `app.py`:
```python
MODEL_PATH = 'best_model.pth'  # or 'optimized_halfunet_physical.pth'
```

### Server Settings
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Allowed File Types
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
```

## 🐛 Troubleshooting

### Model not found
```
❌ Model file not found: best_model.pth
```
**Solution:** Pastikan file `best_model.pth` atau `optimized_halfunet_physical.pth` ada di direktori yang sama dengan `app.py`.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** 
1. Model akan otomatis fallback ke CPU
2. Atau kurangi ukuran gambar input

### Port already in use
```
OSError: [Errno 48] Address already in use
```
**Solution:** 
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

## 🎯 Performance

### Processing Time (RTX 3080):
- 256×256 image: ~50-100ms
- 512×512 image: ~150-200ms
- 1024×1024 image: ~300-500ms

### CPU Mode:
- 2-3x slower than GPU
- Still functional, just takes longer

## 📝 Notes

- Images are automatically resized to 256×256 for processing
- Output is resized back to original dimensions
- Model works best on images with noise similar to training data (SIDD dataset)
- Batch processing not yet implemented (processes one image at a time)

## 🔄 Updates

### Version 1.0.0 (Current)
- Initial release
- Single image processing
- Web interface with drag & drop
- Base64 image encoding for display
- Health check endpoint
- Model info endpoint

## 📞 Support

Jika ada masalah:
1. Check server logs untuk error messages
2. Pastikan semua dependencies terinstall
3. Verifikasi model file ada dan dapat diload
4. Check `/health` endpoint untuk status

## 🙏 Acknowledgments

- Model Architecture: NAFNet (Nonlinear Activation Free Network)
- Dataset: SIDD (Smartphone Image Denoising Dataset)
- Framework: PyTorch + Flask
