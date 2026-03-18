import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pickle
from fastapi import FastAPI, UploadFile, File
import io
import torch.nn.functional as F
import uvicorn

# Define SimpleCNN class (from previous notebook cells)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Global variables for model, transform, and features
model = None
transform = None
device = None
all_reference_features = None
reference_image_paths = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, transform, device, all_reference_features, reference_image_paths

    # Constants (from previous notebook cells)
    INPUT_IMAGE_SIZE = 128
    NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    NORMALIZE_STD = [0.5, 0.5, 0.5]

    # --- FIX: Change MODEL_PATH, FEATURES_SAVE_PATH, PATHS_SAVE_PATH to relative paths for Render deployment ---
    # These files must be uploaded to your GitHub repository alongside main.py
    MODEL_PATH = "model.pth"
    FEATURES_SAVE_PATH = 'all_reference_features.pt'
    PATHS_SAVE_PATH = 'reference_image_paths.pkl'
    # --- END FIX ---

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Loading
    # If model.pth doesn't exist, create a dummy one for demonstration purposes
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}. Creating a dummy model for demonstration.")
        dummy_classes = [str(i) for i in range(10)]
        dummy_num_classes = len(dummy_classes)
        dummy_model_instance = SimpleCNN(num_classes=dummy_num_classes)
        dummy_checkpoint = {
            "classes": dummy_classes,
            "model_state": dummy_model_instance.state_dict(),
        }
        # Ensure the directory exists before saving, but for relative paths, os.path.dirname might be empty
        # If MODEL_PATH is just "model.pth", os.path.dirname returns empty string, os.makedirs will not be needed unless a subdirectory is used.
        # Let's ensure the parent directory exists if it's not the current directory
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(dummy_checkpoint, MODEL_PATH)
        print(f"Dummy model created at {MODEL_PATH}.")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded (classes: {num_classes})")

    # Image Preprocessing Transform
    transform = transforms.Compose([
        transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    print("Image transformation pipeline defined.")

    # Load Reference Features and Paths
    try:
        all_reference_features = torch.load(FEATURES_SAVE_PATH)
        print(f"Reference features loaded from {FEATURES_SAVE_PATH}. Shape: {all_reference_features.shape}")
    except FileNotFoundError:
        print(f"Error: Reference features file not found at {FEATURES_SAVE_PATH}")
        all_reference_features = torch.empty(0, 128) # Initialize as empty tensor with correct feature dimension

    try:
        with open(PATHS_SAVE_PATH, 'rb') as f:
            reference_image_paths = pickle.load(f)
        print(f"Reference image paths loaded from {PATHS_SAVE_PATH}. Count: {len(reference_image_paths)}")
    except FileNotFoundError:
        print(f"Error: Reference image paths file not found at {PATHS_SAVE_PATH}")
        reference_image_paths = [] # Initialize as empty list

    print("API startup initialization complete.")


@app.get("/")
async def read_root():
    return {"message": "Hello World from FastAPI! Model and features loaded."}

# New image upload endpoint
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if model is None or transform is None or device is None or all_reference_features is None or reference_image_paths is None:
        return {"error": "Model or reference data not loaded. Please ensure startup_event ran successfully.", "filename": file.filename}

    try:
        # 1. アップロードされたファイルの内容を読み込み、PIL (Pillow) の Image.open() を使用して画像を開いてください。この際、画像をRGB形式に変換してください。
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. グローバル変数として定義されているtransformオブジェクト（リサイズ、ToTensor、正規化を含む）を読み込んだ画像に適用し、テンソルに変換してください。
        image_tensor = transform(image)

        # 3. 変換されたテンソルにバッチ次元を追加し（unsqueeze(0)）、グローバル変数deviceで指定された適切なデバイス（CPUまたはGPU）に移動させてください。
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # 4. グローバル変数としてロードされているmodelオブジェクトをeval()モードに設定し、torch.no_grad()コンテキスト内で、
        #    前処理済みの画像テンソルをmodel.featuresモジュールに通して特徴量を抽出してください。
        model.eval()
        with torch.no_grad():
            uploaded_image_features = model.features(image_tensor)

        # 5. 抽出された特徴量を1次元ベクトルに整形してください（例: features.view(features.size(0), -1)）。
        uploaded_image_features = uploaded_image_features.view(uploaded_image_features.size(0), -1)

        # 6. コサイン類似度を計算
        # まず、特徴量ベクトルを正規化します
        uploaded_image_features_norm = F.normalize(uploaded_image_features, p=2, dim=1)
        all_reference_features_norm = F.normalize(all_reference_features.to(device), p=2, dim=1)

        # 正規化されたベクトルでは、単なる内積になります
        similarities = torch.matmul(uploaded_image_features_norm, all_reference_features_norm.transpose(0, 1))

        # 最も類似度の高い参照画像を見つける
        best_match_index = torch.argmax(similarities, dim=1).item()
        best_match_similarity = similarities[0, best_match_index].item()
        best_match_path = reference_image_paths[best_match_index]

        print(f"Uploaded image processed: {file.filename}")
        print(f"Extracted features shape: {uploaded_image_features.shape}")
        print(f"Extracted features device: {uploaded_image_features.device}")
        print(f"Best match: {best_match_path}, Similarity: {best_match_similarity:.4f}")

        return {
            "filename": file.filename,
            "message": "Image processed and similarity calculated.",
            "best_match_image": best_match_path,
            "similarity_score": f"{best_match_similarity:.4f}"
        }

    except Exception as e:
        return {"error": f"Error processing image: {e}", "filename": file.filename}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
