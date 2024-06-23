import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Vanishing Gradient Demo")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Network parameters
LAYERS = 5
NODES_PER_LAYER = 4
NODE_RADIUS = 20
LAYER_SPACING = 150

# Gradient decay factor
DECAY_FACTOR = 0.7

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gradient = 1.0
        self.activation = random.random()

    def draw(self, screen):
        color = (int((1 - self.gradient) * 255), int(self.gradient * 255), 0)
        pygame.draw.circle(screen, color, (self.x, self.y), NODE_RADIUS)
        font = pygame.font.Font(None, 20)
        text = font.render(f"{self.activation:.2f}", True, BLACK)
        screen.blit(text, (self.x - 15, self.y - 10))

class Network:
    def __init__(self):
        self.rebuild()

    def rebuild(self):
        self.nodes = []
        for layer in range(LAYERS):
            layer_nodes = []
            for node in range(NODES_PER_LAYER):
                x = 100 + layer * LAYER_SPACING
                y = 100 + node * (HEIGHT - 200) // (NODES_PER_LAYER - 1)
                layer_nodes.append(Node(x, y))
            self.nodes.append(layer_nodes)

    def draw(self, screen):
        for i, layer in enumerate(self.nodes):
            for j, node in enumerate(layer):
                node.draw(screen)
                if i < len(self.nodes) - 1:
                    for next_node in self.nodes[i+1]:
                        pygame.draw.line(screen, BLACK, (node.x, node.y), (next_node.x, next_node.y), 1)

    def update_gradients(self):
        for layer in reversed(range(LAYERS)):
            for node in self.nodes[layer]:
                if layer == LAYERS - 1:
                    node.gradient = 1.0
                else:
                    node.gradient *= DECAY_FACTOR

    def forward_pass(self):
        for layer in self.nodes:
            for node in layer:
                node.activation = random.random()

# Create network
network = Network()

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        font = pygame.font.Font(None, 24)
        text = font.render(self.text, True, BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
add_layer_btn = Button(10, HEIGHT - 50, 100, 40, "Add Layer", GREEN)
remove_layer_btn = Button(120, HEIGHT - 50, 120, 40, "Remove Layer", RED)
train_btn = Button(250, HEIGHT - 50, 100, 40, "Train", BLUE)

# Main game loop
running = True
clock = pygame.time.Clock()
training = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if add_layer_btn.is_clicked(event.pos):
                LAYERS += 1
                network.rebuild()
            elif remove_layer_btn.is_clicked(event.pos):
                if LAYERS > 2:
                    LAYERS -= 1
                    network.rebuild()
            elif train_btn.is_clicked(event.pos):
                training = not training

    # Clear screen
    screen.fill(WHITE)

    # Draw network
    network.draw(screen)

    # Update gradients and perform forward pass if training
    if training:
        network.update_gradients()
        network.forward_pass()

    # Draw buttons
    add_layer_btn.draw(screen)
    remove_layer_btn.draw(screen)
    train_btn.draw(screen)

    # Draw explanatory text
    font = pygame.font.Font(None, 24)
    text = font.render("Vanishing Gradient Interactive Demo", True, BLUE)
    screen.blit(text, (10, 10))
    text = font.render(f"Layers: {LAYERS}, Decay Factor: {DECAY_FACTOR:.2f}", True, BLUE)
    screen.blit(text, (10, 40))
    text = font.render("Green to Red gradient shows vanishing effect", True, BLUE)
    screen.blit(text, (10, 70))

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(5)  # Update 5 times per second for visibility

pygame.quit()