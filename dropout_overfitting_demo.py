import pygame
import random

# Initialize Pygame
pygame.init()
width, height = 1000, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Dropout in Neural Networks - Interactive Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Fonts
font = pygame.font.Font(None, 32)

# Neural network structure
layers = [4, 6, 5, 3]  # Number of neurons in each layer
neurons = []
connections = []

# Initialize neurons
for i, layer_size in enumerate(layers):
    layer = []
    for j in range(layer_size):
        x = 200 + i * 200
        y = 150 + j * (300 / (layer_size - 1))
        layer.append((x, y))
    neurons.append(layer)

# Initialize connections
for i in range(len(layers) - 1):
    layer_connections = []
    for neuron1 in neurons[i]:
        for neuron2 in neurons[i+1]:
            layer_connections.append((neuron1, neuron2))
    connections.append(layer_connections)

# Dropout settings
dropout_rate = 0.5
active_neurons = [list(range(len(layer))) for layer in neurons]

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

# Slider class
class Slider:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_rect = pygame.Rect(x, y, 20, height)
        self.value = 0.5

    def draw(self):
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, BLUE, self.handle_rect)

    def update(self, pos):
        if self.rect.collidepoint(pos):
            self.value = (pos[0] - self.rect.x) / self.rect.width
            self.value = max(0, min(1, self.value))
            self.handle_rect.x = self.rect.x + (self.rect.width - self.handle_rect.width) * self.value

# Create buttons and slider
apply_dropout_button = Button(50, 520, 150, 40, "Apply Dropout", GREEN)
reset_button = Button(220, 520, 150, 40, "Reset", (255, 200, 0))
dropout_slider = Slider(420, 530, 200, 20)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if apply_dropout_button.is_clicked(event.pos):
                dropout_rate = dropout_slider.value
                for i in range(1, len(layers) - 1):  # Skip input and output layers
                    layer_size = len(neurons[i])
                    num_dropout = int(layer_size * dropout_rate)
                    active_neurons[i] = random.sample(range(layer_size), layer_size - num_dropout)
            elif reset_button.is_clicked(event.pos):
                active_neurons = [list(range(len(layer))) for layer in neurons]
            dropout_slider.update(event.pos)
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            dropout_slider.update(event.pos)

    # Drawing
    screen.fill(WHITE)

    # Draw connections
    for i, layer_connections in enumerate(connections):
        for start, end in layer_connections:
            start_layer = i
            end_layer = i + 1
            start_index = neurons[start_layer].index(start)
            end_index = neurons[end_layer].index(end)
            if start_index in active_neurons[start_layer] and end_index in active_neurons[end_layer]:
                pygame.draw.line(screen, GRAY, start, end, 1)

    # Draw neurons
    for i, layer in enumerate(neurons):
        for j, (x, y) in enumerate(layer):
            color = BLUE if j in active_neurons[i] else RED
            pygame.draw.circle(screen, color, (x, y), 15)

    # Draw buttons and slider
    apply_dropout_button.draw()
    reset_button.draw()
    dropout_slider.draw()

    # Display dropout rate
    dropout_text = font.render(f"Dropout Rate: {dropout_slider.value:.2f}", True, BLACK)
    screen.blit(dropout_text, (420, 490))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()