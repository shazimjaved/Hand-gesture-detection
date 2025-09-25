✋ Hand Gesture Recognition

A real-time Hand Gesture Detection system built using OpenCV and MediaPipe.
It detects multiple gestures such as Fist, Open Palm, Thumbs Up, Thumbs Down, Peace Sign, Index Pointing, and Rock 🤘 with live webcam feed or video input.

🚀 Features

🎥 Works with Webcam or Video file input

Detects gestures:

✊ Fist

🖐 Open Palm

👍 Thumbs Up

👎 Thumbs Down

☝️ Index Pointing

✌ Peace Sign

🤘 Rock

⚡ Real-time detection using MediaPipe Hands

📊 Confidence score for each prediction

🧩 Easy to extend for new gestures

🛠 Installation

Clone this repository:

git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition


Create and activate a virtual environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\Activate.ps1      # On Windows


Install dependencies:

pip install -r requirements.txt

▶️ Usage
Run with webcam
python main.py

Run with a video file
python main.py --video sample_video.mp4

🤝 Contributing

Pull requests are welcome!
If you’d like to add more gestures or improve accuracy, feel free to open an issue or PR.

📜 License

This project is licensed under the MIT License
