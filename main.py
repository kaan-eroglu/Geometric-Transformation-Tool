import numpy as np
import matplotlib.pyplot as plt

class GeometricTransformer:
    def __init__(self, points):
        """
        Initialize with a set of points.
        Points should be an (N, 2) or (N, 3) array.
        """
        points = np.asarray(points)
        if points.shape[1] == 2:
            # Convert to homogeneous coordinates (N, 3) by adding a column of 1s
            self.points = np.hstack([points, np.ones((points.shape[0], 1))])
        elif points.shape[1] == 3:
            self.points = points.copy()
        else:
            raise ValueError("Points must be an (N,2) or (N,3) array.")
        
        self.original_points = self.points.copy()
        
        # History of points for step-by-step visualization
        self.history = [self.original_points.copy()]
        self.transform_names = ["Original"]
        self.compound_matrix = np.eye(3)

    def rotate(self, angle_degrees):
        """Rotate points by angle in degrees."""
        theta = np.radians(angle_degrees)
        c, s = np.cos(theta), np.sin(theta)
        # Rotation matrix (2D rotation around origin extended to 3x3)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        self._apply_transform(R, f"Rotate {angle_degrees}°")
        return self

    def scale(self, sx, sy):
        """Scale points by sx and sy."""
        # Scaling matrix
        S = np.array([
            [sx,  0, 0],
            [ 0, sy, 0],
            [ 0,  0, 1]
        ])
        self._apply_transform(S, f"Scale ({sx}, {sy})")
        return self

    def reflect(self, axis):
        """Reflect points across 'x', 'y', or 'y=x'."""
        if axis == 'x':
            F = np.array([
                [1,  0, 0],
                [0, -1, 0],
                [0,  0, 1]
            ])
            name = "Reflect x-axis"
        elif axis == 'y':
            F = np.array([
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 1]
            ])
            name = "Reflect y-axis"
        elif axis == 'y=x':
            F = np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])
            name = "Reflect y=x"
        else:
            raise ValueError("Axis must be 'x', 'y', or 'y=x'.")
        
        self._apply_transform(F, name)
        return self

    def _apply_transform(self, matrix, name):
        """Helper to apply a 3x3 transformation matrix to the points."""
        # Update compound matrix
        self.compound_matrix = matrix @ self.compound_matrix
        
        # Post-multiply since points are row vectors (N, 3): P_transformed = P @ M.T
        self.points = self.points @ matrix.T
        
        # Record history
        self.history.append(self.points.copy())
        self.transform_names.append(name)

    def get_points(self):
        """Return the current 2D points (N, 2)."""
        return self.points[:, :2]

    def plot_side_by_side(self, save_path=None):
        """Plot the original and the final transformed shape side-by-side."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original Shape
        ax = axes[0]
        pts = self.history[0]
        pts_closed = np.vstack([pts, pts[0]]) # Close the polygon
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=8)
        ax.fill(pts_closed[:, 0], pts_closed[:, 1], color='dodgerblue', alpha=0.3)
        ax.set_title(self.transform_names[0], fontsize=14, fontweight='bold')
        self._set_plot_limits(ax, pts)

        # Final Transformed Shape
        ax = axes[1]
        pts = self.history[-1]
        pts_closed = np.vstack([pts, pts[0]]) 
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], marker='s', linestyle='-', color='crimson', linewidth=2, markersize=8)
        ax.fill(pts_closed[:, 0], pts_closed[:, 1], color='crimson', alpha=0.3)
        
        title = "Final: " + " -> ".join(self.transform_names[1:]) if len(self.transform_names) > 1 else "No Transformations"
        # Wrap long titles if necessary
        import textwrap
        title = "\n".join(textwrap.wrap(title, width=40))
        ax.set_title(title, fontsize=12)
        self._set_plot_limits(ax, pts)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved side-by-side plot to {save_path}")
        else:
            plt.show()

    def plot_step_by_step(self, save_path=None):
        """Plot the step-by-step transition of the shape."""
        num_steps = len(self.history)
        fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
        if num_steps == 1:
            axes = [axes]
        
        # Determine global limits for consistent scaling across subplots
        all_pts = np.vstack(self.history)
        min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
        min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
        
        padding_x = max(1.0, (max_x - min_x) * 0.2)
        padding_y = max(1.0, (max_y - min_y) * 0.2)
        
        for i, (ax, pts, name) in enumerate(zip(axes, self.history, self.transform_names)):
            pts_closed = np.vstack([pts, pts[0]])
            color = plt.cm.viridis(i / max(1, num_steps - 1))
            
            ax.plot(pts_closed[:, 0], pts_closed[:, 1], marker='o', linestyle='-', color=color, linewidth=2)
            ax.fill(pts_closed[:, 0], pts_closed[:, 1], color=color, alpha=0.3)
            ax.set_title(name, fontsize=12)
            
            # Use global limits to show actual position change
            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(0, color='black', linewidth=1.5) # x-axis
            ax.axvline(0, color='black', linewidth=1.5) # y-axis
            ax.set_aspect('equal')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved step-by-step plot to {save_path}")
        else:
            plt.show()

    def _set_plot_limits(self, ax, pts):
        """Helper to set consistent and aesthetic limits for an individual plot."""
        min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
        padding_x = max(1.0, (max_x - min_x) * 0.2)
        padding_y = max(1.0, (max_y - min_y) * 0.2)
        
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
        
        # Emphasize axes
        ax.axhline(0, color='black', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')


def main():
    print("=== Interactive Geometric Transformation Tool ===")
    
    # Simple predefined shapes
    shapes = {
        '1': ("Triangle", np.array([[1, 1], [4, 1], [2.5, 4]])),
        '2': ("Square",   np.array([[1, 1], [3, 1], [3, 3], [1, 3]])),
    }

    print("\nSelect a base shape:")
    print("1. Triangle")
    print("2. Square")
    print("3. Custom (enter points manually)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice in shapes:
        name, points = shapes[choice]
        print(f"\nSelected {name}:\n{points}")
    elif choice == '3':
        points_list = []
        print("\nEnter points one by one as 'x y'. Enter 'done' when finished.")
        while True:
            pt_str = input("Point (x y): ").strip()
            if pt_str.lower() == 'done':
                break
            try:
                x, y = map(float, pt_str.split())
                points_list.append([x, y])
            except ValueError:
                print("Invalid format. Please enter as 'x y'.")
        if len(points_list) < 3:
            print("At least 3 points are required to form a polygon. Using default Triangle.")
            points = shapes['1'][1]
        else:
            points = np.array(points_list)
            print(f"\nCustom Polygon Points:\n{points}")
    else:
        print("Invalid choice. Defaulting to Triangle.")
        points = shapes['1'][1]

    transformer = GeometricTransformer(points)
    
    while True:
        print("\nChoose a transformation to apply:")
        print("1. Rotate")
        print("2. Scale")
        print("3. Reflect")
        print("4. Finish and View Results")
        
        t_choice = input("Enter choice (1/2/3/4): ").strip()
        
        if t_choice == '4':
            break
        elif t_choice == '1':
            try:
                angle = float(input("Enter rotation angle in degrees: "))
                transformer.rotate(angle)
                print(f"Applied rotation by {angle} degrees.")
            except ValueError:
                print("Invalid angle. Must be a number.")
        elif t_choice == '2':
            try:
                params = input("Enter scale factors 'sx sy' (e.g., '1.5 0.5'): ").split()
                sx, sy = float(params[0]), float(params[1])
                transformer.scale(sx, sy)
                print(f"Applied scaling by ({sx}, {sy}).")
            except (ValueError, IndexError):
                print("Invalid scale factors. Must be two numbers separated by a space.")
        elif t_choice == '3':
            axis = input("Enter axis of reflection ('x', 'y', or 'y=x'): ").strip().lower()
            if axis in ['x', 'y', 'y=x']:
                transformer.reflect(axis)
                print(f"Applied reflection across {axis}-axis.")
            else:
                print("Invalid axis.")
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

    final_points = transformer.get_points()
    print("\nFinal Transformed Points:")
    print(np.round(final_points, 3))

    print("\nVisualization:")
    print("1. Side-by-side comparison")
    print("2. Step-by-step transition")
    vis_choice = input("Enter visualization style (1/2): ").strip()

    print("\nOpening visualization... Close the plot window to exit.")
    if vis_choice == '2':
        transformer.plot_step_by_step()
    else:
        transformer.plot_side_by_side()

if __name__ == "__main__":
    main()
