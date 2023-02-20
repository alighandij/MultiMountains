import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from gym import spaces

from sidechannels.custom_mountaincar import AngleModifiedDiscreteMountainCar


class MultiMountainsEnv:
    def __init__(
        self,
        angles: tuple[int],
        gravity: float = 9.8,
        force: float = 4,
        dt: float = 0.04,  # FPS = 25 -> dt = 1 / 25
        delta: float = 1e-12,
    ) -> None:
        self.g = gravity
        self.F = force

        self.dt = dt
        self.delta = delta
        self.angles = angles
        self.points = np.array(self.calc_points(angles))

        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.f = CubicSpline(self.x, self.y, bc_type="clamped")

        self.x_curve = np.linspace(self.x.min(), self.x.max(), 1024)
        self.y_curve = self.f(self.x_curve)

        self.ax = None
        self.fig = None

        self.low = np.array([self.x.min(), -25])
        self.high = np.array([self.x.max(), +25])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32)

        self.counter = 0

    def df(self, x: float):
        f = self.f
        d = self.delta
        return (f(x + d) - f(x)) / d

    def reset(self):
        self.counter = 0
        x = np.linspace(self.x[0], self.x[2], 1024)
        idx = np.argmin(self.f(x))
        self.state = (x[idx], 0)
        return np.array(self.state)

    def get_radius(self):
        y = self.y_curve
        return abs(y.max() - y.min()) / 25

    def init_render(self):
        if not ((self.fig is None) and (self.ax is None)):
            return
        plt.ion()

        x = self.x_curve
        y = self.y_curve
        r = self.get_radius()

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.set_window_title("Multi Mountains")
        self.fig.tight_layout()
        self.ax.scatter(self.x, self.y, c="g")
        self.ax.plot(x, y, c="k")
        # self.ax.plot(x, self.df(y), c="r")

        self.ax.set_xlim(x.min(), x.max())
        # self.ax.set_xticks([])

        # self.ax.set_ylim(y.min() - r * 2, y.max() + r * 2)
        # self.ax.set_yticks([])
        self.ax.grid()
        self.ball = Circle((0, 0), r, zorder=2, color="red")
        self.ax.add_patch(self.ball)

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
        At = np.dot([0, -self.g], [Tx, Ty])
        v = v + At * self.dt + (action - 1) * self.F
        x = x + v * self.dt * Tx

        if x < self.x.min():
            x = self.x.min()
            v = 0

        self.counter += 1
        self.state = (x, v)

        return np.array(self.state), self.reward(), self.done(), {}

    def done(self):
        return self.counter == 500 or self.is_goal_reached()

    def reward(self):
        return -1

    def is_goal_reached(self):
        return self.state[0] >= self.x[-1]

    def h(self, x: float):
        return 0.45 * np.sin(3 * x) + 0.55

    def calc_height(self, a: float):
        b_h = self.h(-np.pi / 6)
        g_h = self.h(0.5)
        angle = np.arctan((g_h - b_h) / 1.1) + 2 * a * np.pi / 360
        return b_h + np.tan(angle) * 1.1

    def get_env_points(self, angle: float) -> tuple[tuple[int]]:
        x1 = -1.2
        x2 = -np.pi / 6
        return (
            (x1, self.h(x1)),
            (x2, self.h(x2)),
            (0.5, self.calc_height(angle))
        )
        
    def calc_points(self, angles: tuple[float]) -> tuple:
        points = []
        for i, angle in enumerate(angles):
            pts = self.get_env_points(angle)
            idx = len(points) - 1
            for j, (x, y) in enumerate(pts):
                if i == 0:
                    points.append((x, y))
                    continue
                # I > 1
                if j == 0:
                    continue
                lst_x = points[idx][0]
                lst_y = points[idx][1]
                off_x = lst_x - pts[0][0]
                off_y = lst_y - pts[0][1]
                points.append((x + off_x, y + off_y))
                
                
        return points         

