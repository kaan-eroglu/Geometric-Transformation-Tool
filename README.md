# Geometric-Transformation-Tool
A **Python-based interactive tool** for applying 2D geometric transformations to polygons using **NumPy** for vectorized matrix operations and **Matplotlib** for visualization — served through a clean **Flask web interface**.

![Web UI](static/screenshots/demo_ui.png)

---

## ✨ Features

- **Rotation** — Rotate any shape by a user-specified angle (degrees)
- **Scaling** — Scale independently along X and Y axes
- **Reflection** — Reflect across the X-axis, Y-axis, or the line Y = X
- **Transformation Chaining** — Queue multiple transformations and apply them in sequence
- **Live Visualization** — Side-by-side or step-by-step plots rendered instantly in the browser
- **Multiple Shapes** — Triangle, Square, Pentagon, Star, or define your own custom points
- **Coordinate Table** — View the exact numerical output for every transformed vertex

![Visualization Demo](static/screenshots/demo_side_by_side.png)

---

## 🧮 Mathematical Foundation

All transformations are implemented using **3×3 homogeneous matrices** applied via vectorized NumPy matrix multiplication:

| Transformation | Matrix |
|---|---|
| Rotation by θ | `[[cos θ, -sin θ, 0], [sin θ, cos θ, 0], [0, 0, 1]]` |
| Scaling (sx, sy) | `[[sx, 0, 0], [0, sy, 0], [0, 0, 1]]` |
| Reflect X-axis | `[[1, 0, 0], [0, -1, 0], [0, 0, 1]]` |
| Reflect Y-axis | `[[-1, 0, 0], [0, 1, 0], [0, 0, 1]]` |
| Reflect Y=X | `[[0, 1, 0], [1, 0, 0], [0, 0, 1]]` |

Points are stored as `(N, 3)` homogeneous coordinate arrays. Compound transformations are accumulated via matrix multiplication: `P' = P @ M.T`.

---

## 🗂 Project Structure

```
geometric-transformation-tool/
├── main.py              # GeometricTransformer class (core math engine)
├── app.py               # Flask web server + rendering logic
├── static/
│   ├── index.html       # Interactive web UI
│   └── screenshots/     # Demo screenshots
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kaan-eroglu/geometric-transformation-tool.git
cd geometric-transformation-tool
```

### 2. Create a virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the web app
```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5050**

> ℹ️ `127.0.0.1:5050` is a **local development address** — it only works on your own machine while `app.py` is running. It is not a public website link.

### 4. (Optional) Run the CLI version
```bash
python main.py
```

---

## 🖥 How to Use the Web Interface

1. **Choose a shape** — Triangle, Square, Pentagon, Star, or enter custom coordinates
2. **Build a transformation queue** — Select Rotate / Scale / Reflect, fill in parameters, click **+ Add to Queue**
3. **Select view mode** — Side by Side or Step by Step
4. **Click ▶ Apply & Visualize** — The plot and transformed coordinate table appear instantly

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| **NumPy** | Vectorized matrix × vector operations |
| **Matplotlib** | Plot generation (non-interactive `Agg` backend) |
| **Flask** | Lightweight web server + REST API |
| **HTML / CSS / JS** | Interactive dark-mode frontend (no frameworks) |

---

## 📄 License

MIT License — free to use, modify, and distribute.
