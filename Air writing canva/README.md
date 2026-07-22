# 🖌️🎮 Air Motion Studio (Canvas & Air XO Game)

An ultra-futuristic AI computer vision **Air Motion Studio** powered by **Flask**, **MediaPipe**, **OpenCV**, and **HTML5 Canvas**.

Draw, write text, create neon art, and play **Air XO (Tic-Tac-Toe)** in the air using hand motion gestures tracked via your webcam!

---

## 🌟 What's New & Key Features

- 🎮 **Air XO (Tic-Tac-Toe) Game Mode**:
  - 🤖 **Player vs Unbeatable AI** (Minimax Algorithm) / **Easy AI**.
  - 👥 **2 Player Local Air Turn-Based Mode**.
  - ⏱️ **Dwell Lock-in System**: Target grid cells in the air; holding your index finger for 0.8s fills an SVG radial progress ring and confirms your move!
  - 🏆 **Celebration Modal & Scoreboard HUD**: Live score tracking, winning strike animations, and trophy winner modal.
- 🎨 **Next-Gen Cyber Glassmorphic UI**:
  - Ambient HSL glowing particle background mesh.
  - Top tab navigation (`Air Canvas Studio` vs `Air XO Game`).
  - Left control drawer with tabbed panels (Tools/Colors, XO Config, Camera/Vision, Gesture Guide).
  - Web Audio API Sound FX (Move clicks, win fanfares, clear sounds).
- ✌️ **Gesture Controls**:
  - ☝️ **1 Finger Up (Index)**: Air Drawing mode / Target XO cell.
  - ⏱️ **1 Sec Dwell Hold**: Confirm XO Grid Move.
  - ✌️ **2 Fingers Up (Index + Middle)**: Pointer / Hover mode.
  - 🤟 **3 Fingers Up (Index + Middle + Ring)**: Clear Canvas / Reset XO Board.
  - 🖐️ **Open Palm**: Pause drawing / Move hand neutral.

---

## 🚀 How to Run

### 1. Run the Flask Web Application
Open your terminal inside the project directory:

```cmd
C:\Users\NAVEEN\AppData\Local\Programs\Python\Python312\python.exe app.py
```
*or*
```cmd
py app.py
```

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## ⌨️ Shortcuts & Controls

- **Tab Switcher**: Toggle between `Air Canvas Studio` and `Air XO Game`.
- **Keyboard**:
  - `C`: Clear Board / Canvas
  - `Ctrl + Z`: Undo
  - `Ctrl + Y`: Redo
  - `E`: Toggle Eraser
  - `S`: Save Artwork PNG
