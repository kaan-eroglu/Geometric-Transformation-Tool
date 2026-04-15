import io
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import textwrap
from flask import Flask, request, jsonify, send_from_directory

# Import the GeometricTransformer class from main.py
from main import GeometricTransformer

app = Flask(__name__, static_folder='static')

SHAPES = {
    'triangle': np.array([[1, 1], [4, 1], [2.5, 4]], dtype=float),
    'square':   np.array([[1, 1], [4, 1], [4, 4], [1, 4]], dtype=float),
    'pentagon': np.array([
        [np.cos(2*np.pi*i/5)*2 + 2.5, np.sin(2*np.pi*i/5)*2 + 2.5]
        for i in range(5)
    ], dtype=float),
    'star': np.array([
        [2.5, 4.5], [2.9, 3.2], [4.2, 3.2], [3.2, 2.4],
        [3.6, 1.1], [2.5, 1.9], [1.4, 1.1], [1.8, 2.4],
        [0.8, 3.2], [2.1, 3.2]
    ], dtype=float),
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/transform', methods=['POST'])
def transform():
    data = request.json

    shape_choice = data.get('shape', 'triangle')
    transformations = data.get('transformations', [])
    view_mode     = data.get('view_mode', 'side_by_side')
    custom_points = data.get('custom_points', None)

    # Resolve base shape
    if shape_choice == 'custom' and custom_points:
        try:
            points = np.array(custom_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                raise ValueError
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid custom points. Need at least 3 [x, y] pairs.'}), 400
    elif shape_choice in SHAPES:
        points = SHAPES[shape_choice].copy()
    else:
        points = SHAPES['triangle'].copy()

    transformer = GeometricTransformer(points)

    # Apply each transformation in the list
    for t in transformations:
        kind = t.get('type')
        try:
            if kind == 'rotate':
                transformer.rotate(float(t['angle']))
            elif kind == 'scale':
                transformer.scale(float(t['sx']), float(t['sy']))
            elif kind == 'reflect':
                transformer.reflect(t['axis'])
        except (KeyError, ValueError) as e:
            return jsonify({'error': f'Bad transformation params: {e}'}), 400

    # Render the plot
    try:
        fig = _render_figure(transformer, view_mode)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        return jsonify({'error': f'Rendering error: {e}'}), 500

    final_pts = np.round(transformer.get_points(), 3).tolist()
    steps = transformer.transform_names

    return jsonify({'image': img_b64, 'points': final_pts, 'steps': steps})


def _render_figure(transformer: GeometricTransformer, view_mode: str):
    DARK_BG   = '#0f1117'
    PANEL_BG  = '#1a1d27'
    GRID_CLR  = '#2a2d3e'
    AXIS_CLR  = '#4a4d6a'
    ACCENT1   = '#7c6af7'   # purple
    ACCENT2   = '#f76c6a'   # red-coral
    TEXT_CLR  = '#e2e4f0'

    plt.rcParams.update({
        'text.color': TEXT_CLR,
        'axes.labelcolor': TEXT_CLR,
        'xtick.color': TEXT_CLR,
        'ytick.color': TEXT_CLR,
    })

    num_steps = len(transformer.history)

    if view_mode == 'step_by_step':
        n_cols = num_steps
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))
        fig.patch.set_facecolor(DARK_BG)
        if n_cols == 1:
            axes = [axes]

        all_pts = np.vstack(transformer.history)
        gx0, gx1 = all_pts[:,0].min(), all_pts[:,0].max()
        gy0, gy1 = all_pts[:,1].min(), all_pts[:,1].max()
        px = max(1.2, (gx1-gx0)*0.25)
        py = max(1.2, (gy1-gy0)*0.25)

        palette = [plt.cm.cool(i / max(1, num_steps - 1)) for i in range(num_steps)]

        for i, (ax, pts, name) in enumerate(zip(axes, transformer.history,
                                                  transformer.transform_names)):
            ax.set_facecolor(PANEL_BG)
            pts_c = np.vstack([pts, pts[0]])
            color = palette[i]
            ax.fill(pts_c[:,0], pts_c[:,1], color=color, alpha=0.35)
            ax.plot(pts_c[:,0], pts_c[:,1], '-o', color=color,
                    linewidth=2, markersize=7, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=1.5)
            ax.set_title(name, color=TEXT_CLR, fontsize=11, fontweight='bold', pad=8)
            ax.set_xlim(gx0-px, gx1+px); ax.set_ylim(gy0-py, gy1+py)
            ax.set_aspect('equal')
            ax.axhline(0, color=AXIS_CLR, linewidth=1.2)
            ax.axvline(0, color=AXIS_CLR, linewidth=1.2)
            ax.grid(True, color=GRID_CLR, linewidth=0.8, linestyle='--')
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

    else:  # side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor(DARK_BG)

        # collect limits from BOTH panels together
        all_pts = np.vstack([transformer.history[0], transformer.history[-1]])
        gx0, gx1 = all_pts[:,0].min(), all_pts[:,0].max()
        gy0, gy1 = all_pts[:,1].min(), all_pts[:,1].max()
        px = max(1.2, (gx1-gx0)*0.25)
        py = max(1.2, (gy1-gy0)*0.25)

        for ax, pts, color, marker, title in [
            (axes[0], transformer.history[0],  ACCENT1, 'o', 'Original'),
            (axes[1], transformer.history[-1], ACCENT2, 's', 'Transformed'),
        ]:
            ax.set_facecolor(PANEL_BG)
            pts_c = np.vstack([pts, pts[0]])
            ax.fill(pts_c[:,0], pts_c[:,1], color=color, alpha=0.3)
            ax.plot(pts_c[:,0], pts_c[:,1], linestyle='-', marker=marker,
                    color=color, linewidth=2.5, markersize=9,
                    markerfacecolor='white', markeredgecolor=color,
                    markeredgewidth=1.8)
            ax.set_title(title, color=TEXT_CLR, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlim(gx0-px, gx1+px); ax.set_ylim(gy0-py, gy1+py)
            ax.set_aspect('equal')
            ax.axhline(0, color=AXIS_CLR, linewidth=1.2)
            ax.axvline(0, color=AXIS_CLR, linewidth=1.2)
            ax.grid(True, color=GRID_CLR, linewidth=0.8, linestyle='--')
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

        transform_label = ' → '.join(transformer.transform_names[1:]) or 'None'
        transform_label = '\n'.join(textwrap.wrap(transform_label, width=70))
        fig.suptitle(f'Transformations: {transform_label}',
                     color=TEXT_CLR, fontsize=11, y=1.01, fontweight='bold')

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=5050)
