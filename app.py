
from flask import Flask, request, render_template
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from model import SimpleCNN 

app = Flask(__name__)

model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    if file:
        img = Image.open(file.stream)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
             transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            predicted_class = get_class_name(predicted.item())
        return render_template('result.html', classification=predicted_class)

def get_class_name(class_id):
    classes = ['Apta', 'Indian Rubber Tree', 'Karanj','Kashid', 'Nilgiri', 'Pimpal', 'Sita Ashok', 'Sonmohar','Vad', 'Vilayati Chinch' ]
    return classes[class_id]

if __name__ == '__main__':
    app.run(debug=True)
