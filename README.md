# LipSync AI

LipSync AI is a deep learning pipeline for generating speech audio from lip movements in video. It includes modules for face detection, lip region extraction, visual and temporal feature extraction, speech synthesis, and vocoding.

## Features
- Face and lip region detection using MediaPipe
- Visual feature extraction with ResNet
- Temporal encoding with LSTM
- Speech synthesis and vocoder modules
- Modular, extensible codebase

## Project Structure
- `src/` - Source code
- `data/` - Data directories (raw, processed, samples)
- `models/` - Pretrained and trained model weights
- `config.yaml` - Configuration file
- `requirements.txt` - Python dependencies

## Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd lipsync_ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or prepare your data and place it in `data/raw/`.
4. (Optional) Download pretrained models and place them in `models/`.

## Usage
Run the main pipeline on a sample video:
```bash
python src/main.py
```

## Training
To train the model, use:
```bash
python src/training/train.py
```

## Evaluation
To evaluate the model, use:
```bash
python src/training/evaluate.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 
