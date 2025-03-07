import pygame
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Activation Function Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Fonts
font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 36)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Function information
functions = [relu, sigmoid, tanh]
function_names = ["ReLU", "Sigmoid", "Tanh"]
function_explanations = [
    "ReLU: f(x) = max(0, x)\nOutputs the input if positive, else 0.\nUsed in hidden layers, allows for sparse activation.",
    "Sigmoid: f(x) = 1 / (1 + e^-x)\nSquashes input to range (0, 1).\nUsed in binary classification output layers.",
    "Tanh: f(x) = (e^x - e^-x) / (e^x + e^-x)\nSquashes input to range (-1, 1).\nOften used in hidden layers of neural networks."
]

current_function_index = 0

# Graph settings
graph_width = 1200
graph_height = 600
x_min, x_max = -10, 10
y_min, y_max = -2, 2

# Convert graph coordinates to screen coordinates
def graph_to_screen(x, y):
    screen_x = (x - x_min) / (x_max - x_min) * graph_width + 50
    screen_y = HEIGHT - 50 - (y - y_min) / (y_max - y_min) * graph_height
    return int(screen_x), int(screen_y)

# Draw the graph
def draw_graph():
    # Draw axes
    origin = graph_to_screen(0, 0)
    pygame.draw.line(screen, BLACK, (50, origin[1]), (WIDTH - 50, origin[1]), 2)
    pygame.draw.line(screen, BLACK, (origin[0], HEIGHT - 50), (origin[0], 50), 2)

    # Draw function
    x = np.linspace(x_min, x_max, 1000)
    y = functions[current_function_index](x)
    points = [graph_to_screen(x[i], y[i]) for i in range(len(x))]
    pygame.draw.lines(screen, BLUE, False, points, 2)

    # Draw axes labels
    for i in range(x_min, x_max + 1, 2):
        pos = graph_to_screen(i, 0)
        pygame.draw.line(screen, BLACK, (pos[0], pos[1] - 5), (pos[0], pos[1] + 5), 2)
        label = font.render(str(i), True, BLACK)
        screen.blit(label, (pos[0] - 10, pos[1] + 10))

    for i in range(y_min, y_max + 1):
        pos = graph_to_screen(0, i)
        pygame.draw.line(screen, BLACK, (pos[0] - 5, pos[1]), (pos[0] + 5, pos[1]), 2)
        label = font.render(str(i), True, BLACK)
        screen.blit(label, (pos[0] - 30, pos[1] - 10))

# Interactive point
interactive_x = 0
def update_interactive_point(mouse_x):
    global interactive_x
    interactive_x = (mouse_x - 50) / graph_width * (x_max - x_min) + x_min
    interactive_x = max(x_min, min(x_max, interactive_x))

def draw_interactive_point():
    x = interactive_x
    y = functions[current_function_index](x)
    pos = graph_to_screen(x, y)
    pygame.draw.circle(screen, RED, pos, 5)
    
    input_text = font.render(f"Input: {x:.2f}", True, BLACK)
    output_text = font.render(f"Output: {y:.2f}", True, BLACK)
    screen.blit(input_text, (50, HEIGHT - 40))
    screen.blit(output_text, (200, HEIGHT - 40))

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                current_function_index = (current_function_index + 1) % len(functions)
        elif event.type == pygame.MOUSEMOTION:
            update_interactive_point(event.pos[0])

    screen.fill(WHITE)

    # Draw title and current function
    title = title_font.render("Activation Function Demo", True, BLACK)
    screen.blit(title, (10, 10))

    function_name = title_font.render(f"Current Function: {function_names[current_function_index]}", True, BLACK)
    screen.blit(function_name, (10, 50))

    # Draw function explanation
    explanation_lines = function_explanations[current_function_index].split('\n')
    for i, line in enumerate(explanation_lines):
        explanation_text = font.render(line, True, BLACK)
        screen.blit(explanation_text, (10, 90 + i * 30))

    # Draw graph and interactive point
    draw_graph()
    draw_interactive_point()

    # Draw instructions
    instruction_text = font.render("Move mouse to change input. Press SPACE to change function.", True, BLACK)
    screen.blit(instruction_text, (50, HEIGHT - 70))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
