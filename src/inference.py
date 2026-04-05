import cv2
import torch
from model import get_model

def infer(image_path, model_path):
    model = get_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (512, 256)) / 255.0

    tensor = torch.tensor(img_resized).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(tensor)['out']
        mask = torch.sigmoid(output).squeeze().numpy()

    mask = (mask > 0.5).astype('uint8') * 255

    return mask
