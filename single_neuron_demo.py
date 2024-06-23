import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Single Neuron Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Neuron parameters
num_inputs = 3
weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
bias = random.uniform(-1, 1)  # Move this line outside of any function
learning_rate = 0.1

# Font
font = pygame.font.Font(None, 24)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron_output(inputs):
    global weights, bias  # Add this line
    weighted_sum = sum(w * i for w, i in zip(weights, inputs)) + bias
    return sigmoid(weighted_sum)

def draw_neuron(x, y, radius):
    pygame.draw.circle(screen, BLUE, (x, y), radius)

def draw_inputs(inputs):
    for i, input_val in enumerate(inputs):
        x = 100
        y = 150 + i * 100
        pygame.draw.circle(screen, RED, (x, y), 20)
        text = font.render(f"Input {i+1}: {input_val:.2f}", True, BLACK)
        screen.blit(text, (x + 30, y - 10))

def draw_weights():
    for i, weight in enumerate(weights):
        x1, y1 = 100, 150 + i * 100
        x2, y2 = 400, 300
        pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 2)
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        text = font.render(f"W{i+1}: {weight:.2f}", True, BLACK)
        screen.blit(text, (mid_x, mid_y))

def draw_output(output):
    x, y = 600, 300
    pygame.draw.circle(screen, GREEN, (x, y), 20)
    text = font.render(f"Output: {output:.2f}", True, BLACK)
    screen.blit(text, (x + 30, y - 10))

def main():
    global weights, bias  # Add this line
    inputs = [random.random() for _ in range(num_inputs)]
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    for i in range(num_inputs):
                        x, y = 100, 150 + i * 100
                        if math.hypot(event.pos[0] - x, event.pos[1] - y) < 20:
                            inputs[i] = random.random()

        screen.fill(WHITE)

        # Draw neuron components
        draw_neuron(400, 300, 30)
        draw_inputs(inputs)
        draw_weights()
        output = neuron_output(inputs)
        draw_output(output)

        # Draw bias
        bias_text = font.render(f"Bias: {bias:.2f}", True, BLACK)
        screen.blit(bias_text, (350, 250))

        # Instructions
        instructions = [
            "Click on input circles to randomize their values",
            "Press 'R' to reset weights and bias",
            "Press 'T' to train the neuron",
        ]
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, BLACK)
            screen.blit(text, (10, 10 + i * 30))

        pygame.display.flip()
        clock.tick(60)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            weights[:] = [random.uniform(-1, 1) for _ in range(num_inputs)]
            bias = random.uniform(-1, 1)
        elif keys[pygame.K_t]:
            target = random.randint(0, 1)
            output = neuron_output(inputs)
            error = target - output
            for i in range(num_inputs):
                weights[i] += learning_rate * error * inputs[i]
            bias += learning_rate * error

    pygame.quit()

if __name__ == "__main__":
    main()