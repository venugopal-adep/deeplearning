import pygame
import numpy as np
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Global Optimization Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Fonts
font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 48)
developer_font = pygame.font.Font(None, 28)

# Function with multiple local minima
def f(x, y):
    return 2 * np.sin(0.5*x) * np.cos(0.5*y) + (x**2 + y**2) / 20

# Gradient of the function
def gradient(x, y):
    dx = np.cos(0.5*x) * np.cos(0.5*y) + x / 10
    dy = -np.sin(0.5*x) * np.sin(0.5*y) + y / 10
    return np.array([dx, dy])

# 3D projection parameters
angle = 0
SCALE = 40
Y_OFFSET = 200  # Add this line to create a vertical offset

# Convert 3D coordinates to 2D screen coordinates
def project(x, y, z):
    x_rot = x * math.cos(angle) - y * math.sin(angle)
    y_rot = x * math.sin(angle) + y * math.cos(angle)
    z_rot = z
    x_proj = x_rot * SCALE + WIDTH // 2
    y_proj = -z_rot * SCALE + HEIGHT // 2 - y_rot * SCALE * 0.3 + Y_OFFSET  # Modified this line
    return int(x_proj), int(y_proj)

# Draw the 3D function surface
def draw_surface():
    for x in range(-20, 21):
        for y in range(-20, 21):
            z = f(x/2, y/2)
            color = get_color(z)
            x1, y1 = project(x/2, y/2, z)
            x2, y2 = project((x+1)/2, y/2, f((x+1)/2, y/2))
            x3, y3 = project(x/2, (y+1)/2, f(x/2, (y+1)/2))
            pygame.draw.line(screen, color, (x1, y1), (x2, y2))
            pygame.draw.line(screen, color, (x1, y1), (x3, y3))

# Get color based on z-value
def get_color(z):
    r = int(max(0, min(255, 128 + z * 50)))
    g = int(max(0, min(255, 128 - abs(z) * 50)))
    b = int(max(0, min(255, 128 - z * 50)))
    return (r, g, b)

# Gradient descent parameters
learning_rate = 0.1
num_iterations = 100
current_pos = np.array([0.0, 0.0])
positions = [current_pos.copy()]
best_pos = current_pos.copy()
best_value = f(*best_pos)

# Gradient descent step
def gradient_descent_step():
    global current_pos, best_pos, best_value
    grad = gradient(current_pos[0], current_pos[1])
    current_pos -= learning_rate * grad
    positions.append(current_pos.copy())
    current_value = f(*current_pos)
    if current_value < best_value:
        best_pos = current_pos.copy()
        best_value = current_value

# Random restart
def random_restart():
    global current_pos, positions
    current_pos = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
    positions = [current_pos.copy()]

# Main loop
running = True
clock = pygame.time.Clock()
iteration = 0
auto_run = False
restarts = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                x, y = (event.pos[0] - WIDTH // 2) / SCALE, (HEIGHT // 2 + Y_OFFSET - event.pos[1]) / SCALE
                x, y = x * math.cos(-angle) - y * math.sin(-angle), x * math.sin(-angle) + y * math.cos(-angle)
                current_pos = np.array([x, y])
                positions = [current_pos.copy()]
                iteration = 0
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                auto_run = not auto_run
            elif event.key == pygame.K_r:
                random_restart()
                iteration = 0
                restarts += 1
            elif event.key == pygame.K_LEFT:
                angle -= 0.1
            elif event.key == pygame.K_RIGHT:
                angle += 0.1

    screen.fill(WHITE)

    # Draw the 3D surface
    draw_surface()

    # Draw gradient descent path
    for i in range(1, len(positions)):
        start = project(positions[i-1][0], positions[i-1][1], f(*positions[i-1]))
        end = project(positions[i][0], positions[i][1], f(*positions[i]))
        pygame.draw.line(screen, RED, start, end, 3)

    # Draw current position
    current_z = f(*current_pos)
    current_screen_pos = project(current_pos[0], current_pos[1], current_z)
    pygame.draw.circle(screen, BLUE, current_screen_pos, 5)

    # Draw best position
    best_screen_pos = project(best_pos[0], best_pos[1], best_value)
    pygame.draw.circle(screen, YELLOW, best_screen_pos, 7)

    # Display title and developer information
    title = title_font.render("3D Global Optimization Demo", True, BLACK)
    title_rect = title.get_rect(center=(WIDTH // 2, 30))
    screen.blit(title, title_rect)

    developer = developer_font.render("Developed by: Venugopal Adep", True, BLACK)
    developer_rect = developer.get_rect(center=(WIDTH // 2, 70))
    screen.blit(developer, developer_rect)

    # Display information
    info_text = [
        f"Iteration: {iteration}",
        f"Current Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f})",
        f"Current Value: {current_z:.2f}",
        f"Best Position: ({best_pos[0]:.2f}, {best_pos[1]:.2f})",
        f"Best Value: {best_value:.2f}",
        f"Restarts: {restarts}",
        "Spacebar: Toggle auto-run",
        "R: Random restart",
        "Left/Right arrows: Rotate view"
    ]

    for i, text in enumerate(info_text):
        info_surface = font.render(text, True, BLACK)
        screen.blit(info_surface, (WIDTH - 350, 100 + i * 30))

    pygame.display.flip()

    if auto_run:
        if iteration < num_iterations:
            gradient_descent_step()
            iteration += 1
        else:
            random_restart()
            iteration = 0
            restarts += 1

    clock.tick(30)

pygame.quit()