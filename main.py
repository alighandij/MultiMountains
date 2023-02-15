from environment import MultiMountainsEnv
from random import randint
from time import sleep

points = (
    (0, 10),
    (7, 2),
    (15, 10),
    (25, 5),
    (35, 10),
    (40, 0),
    (48, 10)
)

env = MultiMountainsEnv(
    points=points
)

def loop():
    while True:
        a = randint(0, 2)
        env.step(a)
        x, v = env.state
        print(f"A = {a} | X = {x} | Y = {env.f(x)} | V = {v}")
        env.render()

loop()