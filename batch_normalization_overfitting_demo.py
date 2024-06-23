import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()
width, height = 1800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Batch Normalization in Neural Networks - Interactive Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Fonts
title_font = pygame.font.Font(None, 48)
font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 20)

# Neural network structure
layers = [4, 6, 5, 3]  # Number of neurons in each layer
neurons = []
connections = []

# Initialize neurons
for i, layer_size in enumerate(layers):
    layer = []
    for j in range(layer_size):
        x = 400 + i * 350
        y = 200 + j * (400 / (layer_size - 1))
        layer.append((x, y, random.uniform(-1, 1)))  # (x, y, activation)
    neurons.append(layer)

# Initialize connections
for i in range(len(layers) - 1):
    layer_connections = []
    for neuron1 in neurons[i]:
        for neuron2 in neurons[i+1]:
            layer_connections.append((neuron1, neuron2))
    connections.append(layer_connections)

# Batch Normalization
def batch_normalize(layer):
    activations = [neuron[2] for neuron in layer]
    mean = np.mean(activations)
    std = np.std(activations)
    return [(x, y, (a - mean) / (std + 1e-8)) for x, y, a in layer]

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
normalize_button = Button(50, 700, 250, 50, "Apply Normalization", GREEN)
randomize_button = Button(350, 700, 250, 50, "Randomize Values", BLUE)

# Main game loop
running = True
clock = pygame.time.Clock()
normalization_applied = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if normalize_button.is_clicked(event.pos):
                for i in range(1, len(neurons)):  # Skip input layer
                    neurons[i] = batch_normalize(neurons[i])
                normalization_applied = True
            elif randomize_button.is_clicked(event.pos):
                for layer in neurons:
                    for i, (x, y, _) in enumerate(layer):
                        layer[i] = (x, y, random.uniform(-1, 1))
                normalization_applied = False

    # Drawing
    screen.fill(WHITE)

    # Draw title
    title = title_font.render("Batch Normalization in Neural Networks", True, BLACK)
    screen.blit(title, (width // 2 - title.get_width() // 2, 20))

    # Draw developer credit
    credit = font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(credit, (width // 2 - credit.get_width() // 2, 70))

    # Draw connections
    for layer_connections in connections:
        for start, end in layer_connections:
            pygame.draw.line(screen, GRAY, start[:2], end[:2], 1)

    # Draw neurons
    for layer in neurons:
        for x, y, activation in layer:
            color = (
                int(max(0, min(255, 255 * (1 - activation)))),
                0,
                int(max(0, min(255, 255 * (1 + activation))))
            )
            pygame.draw.circle(screen, color, (int(x), int(y)), 20)
            
            # Display activation value
            value_text = small_font.render(f"{activation:.2f}", True, BLACK)
            screen.blit(value_text, (int(x) - 20, int(y) + 25))

    # Draw buttons
    normalize_button.draw()
    randomize_button.draw()

    # Display info
    screen.blit(font.render("Activation Values:", True, BLACK), (50, 100))
    screen.blit(small_font.render("Red: Negative", True, RED), (50, 140))
    screen.blit(small_font.render("Blue: Positive", True, BLUE), (50, 170))
    
    if normalization_applied:
        screen.blit(font.render("Normalization Applied", True, GREEN), (1400, 710))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()