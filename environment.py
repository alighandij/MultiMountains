import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline


class MultiMountainsEnv:
    def __init__(
        self,
        points: tuple[int],
        gravity: float = 9.8,
        force: float = 4,
        dt: float = 0.04,  # FPS = 25 -> dt = 1 / 25
        delta: float = 1e-12,
    ) -> None:
        self.g = gravity
        self.F = force

        self.dt = dt
        self.delta = delta
        self.points = np.array(points)

        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.f = CubicSpline(self.x, self.y, bc_type="clamped")

        self.x_curve = np.linspace(self.x.min(), self.x.max(), 1024)
        self.y_curve = self.f(self.x_curve)

        self.ax = None
        self.fig = None

        self.state = (self.x[1], 0.0)

    def df(self, x: float):
        f = self.f
        d = self.delta
        return (f(x + d) - f(x)) / d

    def init_render(self):
        if not ((self.fig is None) and (self.ax is None)):
            return
        plt.ion()
        
        x = self.x_curve
        y = self.y_curve
        
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.set_window_title("Multi Mountains")
        self.fig.tight_layout()
        
        self.ax.plot(x, y)
        
        self.ax.set_xlim(x.min(), x.max())
        self.ax.set_xticks([])
        
        self.ax.set_ylim(y.min() - 2, y.max() + 2)
        self.ax.set_yticks([])
        
        self.ball = Circle((0, 0), 0.25)
        self.ax.add_patch(self.ball)
        self.ax.patches[0].set_color("red")

    def render(self):
        self.init_render()
        x, _ = self.state
        y = self.f(x)
        self.ball.set_center((x, y))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    def step(self, action: int):
        x, v = self.state
        dy = self.df(x)
        delta_x = self.delta
        delta_y = delta_x * dy
        mag = np.sqrt(delta_x ** 2 + delta_y ** 2)
        Tx = delta_x / mag
        Ty = delta_y / mag
        At = np.dot([0, -self.g], [Tx, Ty]) + (action - 1) * self.F
        v = v + At * self.dt 
        x = x + v * self.dt * Tx
        if x < self.x.min():
            x = self.x.min()
            v = 0

        self.state = (x, v)

        return np.array(self.state), self.reward, self.done(), {}

    def done(self):
        return False

    def reward(self):
        return -1
