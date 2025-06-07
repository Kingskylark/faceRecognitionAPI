# Face Recognition API

A comprehensive FastAPI-based face recognition service using OpenCV for student registration and attendance systems.

## Features

- 🔍 **Face Detection**: Multi-method face detection with enhanced accuracy
- 🎯 **Face Registration**: Register student faces with comprehensive feature extraction
- ⚖️ **Face Comparison**: Compare faces using multiple similarity metrics
- 🔒 **Image Validation**: Validate uploaded images for face detection
- 🌐 **REST API**: Easy integration with web applications
- 🐳 **Docker Support**: Containerized deployment
- 📊 **Multiple Formats**: Support for file uploads and base64 images

## Quick Start

### Method 1: Local Installation

1. **Clone and setup**:
```bash
git clone <your-repo>
cd face-recognition-api
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the server**:
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4. **Access the API**:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Method 2: Docker Deployment

1. **Using Docker Compose (Recommended)**:
```bash
docker-compose up -d
```

2. **Using Docker directly**:
```bash
docker build -t face-recognition-api .
docker run -p 8000:8000 face-recognition-api
```

## API Endpoints

### 🏠 **Root & Health**
- `GET /` - API information
- `GET /health` - Health check

### 👤 **Face Registration**
- `POST /register` - Register face from uploaded file
- `POST /register/base64` - Register face from base64 image

### 🔍 **Face Comparison**
- `POST /compare` - Compare two feature vectors
- `POST /compare/json` - Compare features from JSON strings

### ✅ **Validation & Extraction**
- `POST /validate` - Validate image for face detection
- `POST /extract-features` - Extract features without registration

## Usage Examples

### Python Client
```python
from client import FaceRecognitionClient

client = FaceRecognitionClient("http://localhost:8000")

# Register a face
result = client.register_face_file("student.jpg", "student_001")
print(result)

# Compare faces
comparison = client.compare_faces(features1, features2, threshold=0.6)
print(comparison)
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# Register face
curl -X POST "http://localhost:8000/register" \
  -F "file=@student.jpg" \
  -F "student_id=student_001"

# Compare faces
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "features1": [...],
    "features2": [...],
    "threshold": 0.6
  }'
```

### JavaScript/Web Integration
```javascript
// Register face from file input
async function registerFace(imageFile, studentId) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('student_id', studentId);
    
    const response = await fetch('http://localhost:8000/register', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Compare faces
async function compareFaces(features1, features2) {
    const response = await fetch('http://localhost:8000/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            features1: features1,
            features2: features2,
            threshold: 0.6
        })
    });
    
    return await response.json();
}
```

## Response Format

### Successful Registration
```json
{
  "success": true,
  "message": "Face encoding successful",
  "data": {
    "face_encoding": [...],
    "student_id": "student_001",
    "face_features_json": "[...]",
    "face_region": "{...}",
    "face_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "features_length": 470,
    "validation_message": "Image is valid - 1 face(s) detected",
    "extraction_method": "OpenCV_Enhanced_Features",
    "total_faces_detected": 1
  }
}
```

### Face Comparison
```json
{
  "is_match": true,
  "similarity_score": 0.85,
  "cosine_similarity": 0.87,
  "euclidean_similarity": 0.82,
  "correlation": 0.79,
  "manhattan_similarity": 0.91,
  "euclidean_distance": 0.22,
  "manhattan_distance": 0.10,
  "threshold": 0.6
}
```

## Configuration

### Environment Variables
```bash
# Server configuration
PORT=8000
HOST=0.0.0.0

# Face detection thresholds
FACE_DETECTION_THRESHOLD=0.6
MAX_FACES_ALLOWED=5
MIN_FACE_SIZE=50

# Upload limits
MAX_FILE_SIZE=10MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,bmp
```

### Similarity Threshold Guidelines
- **0.5-0.6**: Lenient matching (may have false positives)
- **0.6-0.7**: Balanced matching (recommended)
- **0.7-0.8**: Strict matching (may miss some matches)
- **0.8+**: Very strict (for high-security applications)

## Integration with PHP

```php
<?php
// Register face
function registerFace($imagePath, $studentId) {
    $url = 'http://localhost:8000/register';
    
    $curl = curl_init();
    curl_setopt_array($curl, [
        CURLOPT_URL => $url,
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => [
            'file' => new CURLFile($imagePath),
            'student_id' => $studentId
        ],
        CURLOPT_RETURNTRANSFER => true,
    ]);
    
    $response = curl_exec($curl);
    curl_close($curl);
    
    return json_decode($response, true);
}

// Compare faces
function compareFaces($features1, $features2, $threshold = 0.6) {
    $url = 'http://localhost:8000/compare/json';
    
    $data = [
        'features1_json' => json_encode($features1),
        'features2_json' => json_encode($features2),
        'threshold' => $threshold
    ];
    
    $curl = curl_init();
    curl_setopt_array($curl, [
        CURLOPT_URL => $url,
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => json_encode($data),
        CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
        CURLOPT_RETURNTRANSFER => true,
    ]);
    
    $response = curl_exec($curl);
    curl_close($curl);
    
    return json_decode($response, true);
}
?>
```

## Performance Optimization

### For High-Volume Applications
1. **Use Docker with resource limits**:
```yaml
services:
  face-recognition-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

2. **Implement caching** for frequently compared faces
3. **Use multiple instances** behind a load balancer
4. **Database integration** for storing features

### Image Guidelines
- **Resolution**: 640x480 to 1920x1080 (optimal)
- **Format**: JPG/PNG (avoid BMP for large files)
- **Lighting**: Good, even lighting
- **Face size**: At least 100x100 pixels
- **Quality**: Avoid blurry or low-quality images

## Troubleshooting

### Common Issues

1. **"No face detected"**:
   - Ensure good lighting
   - Face should be clearly visible
   - Try different angles
   - Check image quality

2. **"Too many faces detected"**:
   - Use images with single person
   - Crop to focus on main subject

3. **Low similarity scores**:
   - Check image quality
   - Ensure consistent lighting
   - Verify same person in both images

### Debug Mode
```bash
# Run with debug logging
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## Security Considerations

1. **Input Validation**: All uploads are validated
2. **File Size Limits**: Configurable upload limits
3. **CORS Configuration**: Configure for production
4. **Rate Limiting**: Consider implementing for production
5. **Authentication**: Add authentication layer as needed

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the troubleshooting section

---

**Note**: This API is designed for educational and development purposes. For production use, consider additional security measures, error handling, and performance optimizations.