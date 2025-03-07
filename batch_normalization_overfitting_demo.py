import pygame
import random
import numpy as np
import math

# Initialize Pygame
pygame.init()
width, height = 1600, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Batch Normalization in Neural Networks - Interactive Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
BLUE = (65, 105, 225)
RED = (220, 20, 60)
GREEN = (34, 139, 34)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)

# Fonts
title_font = pygame.font.SysFont("Arial", 48, bold=True)
font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 18)
info_font = pygame.font.SysFont("Arial", 22)

# Neural network structure
layers = [4, 6, 5, 3]  # Number of neurons in each layer
neurons = []
connections = []
layer_names = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]

# Initialize neurons with better spacing
for i, layer_size in enumerate(layers):
    layer = []
    for j in range(layer_size):
        x = 500 + i * 300
        y = 200 + j * (350 / max(layer_size - 1, 1))
        layer.append((x, y, random.uniform(-2, 2)))  # (x, y, activation)
    neurons.append(layer)

# Initialize connections
for i in range(len(layers) - 1):
    layer_connections = []
    for neuron1 in neurons[i]:
        for neuron2 in neurons[i+1]:
            weight = random.uniform(-1, 1)
            layer_connections.append((neuron1, neuron2, weight))
    connections.append(layer_connections)

# Batch Normalization
def batch_normalize(layer):
    activations = [neuron[2] for neuron in layer]
    mean = np.mean(activations)
    std = np.std(activations) + 1e-8  # Add epsilon for numerical stability
    return [(x, y, (a - mean) / std) for x, y, a in layer], mean, std

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color or tuple(max(0, c - 30) for c in color)
        self.is_hovered = False

    def draw(self):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons with better positioning
normalize_button = Button(50, 800, 250, 50, "Apply Normalization", GREEN)
randomize_button = Button(320, 800, 250, 50, "Randomize Values", BLUE)
reset_button = Button(590, 800, 250, 50, "Reset Demo", RED)

# Stats visualization
def draw_stats_box(x, y, width, height, layer_idx, mean, std, is_normalized):
    pygame.draw.rect(screen, LIGHT_GRAY, (x, y, width, height), border_radius=10)
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2, border_radius=10)
    
    title = info_font.render(f"{layer_names[layer_idx]} Statistics", True, BLACK)
    screen.blit(title, (x + 10, y + 10))
    
    mean_text = small_font.render(f"Mean: {mean:.4f}", True, BLACK)
    std_text = small_font.render(f"Std Dev: {std:.4f}", True, BLACK)
    
    screen.blit(mean_text, (x + 10, y + 40))
    screen.blit(std_text, (x + 10, y + 65))
    
    status = "Normalized" if is_normalized else "Not Normalized"
    color = GREEN if is_normalized else RED
    status_text = small_font.render(f"Status: {status}", True, color)
    screen.blit(status_text, (x + 10, y + 90))

# Draw histogram
def draw_histogram(x, y, width, height, values, title):
    pygame.draw.rect(screen, WHITE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    title_text = small_font.render(title, True, BLACK)
    screen.blit(title_text, (x + width//2 - title_text.get_width()//2, y + 5))
    
    # Histogram bins
    if values:
        bins = 10
        hist, bin_edges = np.histogram(values, bins=bins, range=(-3, 3))
        max_count = max(hist) if max(hist) > 0 else 1
        
        bar_width = (width - 20) / bins
        for i in range(bins):
            bar_height = (hist[i] / max_count) * (height - 40)
            bar_x = x + 10 + i * bar_width
            bar_y = y + height - 20 - bar_height
            
            # Color gradient based on bin position
            color_val = int(255 * (i / bins))
            bar_color = (255 - color_val, 0, color_val)
            
            pygame.draw.rect(screen, bar_color, (bar_x, bar_y, bar_width - 2, bar_height))
        
        # Draw x-axis labels
        pygame.draw.line(screen, BLACK, (x + 10, y + height - 20), (x + width - 10, y + height - 20), 2)
        
        # Draw min and max labels
        min_label = small_font.render("-3", True, BLACK)
        max_label = small_font.render("3", True, BLACK)
        screen.blit(min_label, (x + 10, y + height - 18))
        screen.blit(max_label, (x + width - 20, y + height - 18))
        
        # Draw 0 label
        zero_label = small_font.render("0", True, BLACK)
        screen.blit(zero_label, (x + width//2, y + height - 18))

# Animation parameters
animation_speed = 10
animation_progress = 0
animating = False
animation_source = None
animation_target = None

# Main game loop
running = True
clock = pygame.time.Clock()
normalization_applied = [False] * len(neurons)
layer_means = [0] * len(neurons)
layer_stds = [1] * len(neurons)
normalized_neurons = None
show_info = True

while running:
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if normalize_button.is_clicked(event.pos):
                if not animating:
                    normalized_neurons = []
                    for i in range(len(neurons)):
                        if i > 0:  # Skip input layer
                            norm_layer, mean, std = batch_normalize(neurons[i])
                            normalized_neurons.append(norm_layer)
                            layer_means[i] = mean
                            layer_stds[i] = std
                            normalization_applied[i] = True
                        else:
                            normalized_neurons.append(neurons[i])
                    
                    animation_source = neurons
                    animation_target = normalized_neurons
                    animation_progress = 0
                    animating = True
            
            elif randomize_button.is_clicked(event.pos):
                if not animating:
                    for i, layer in enumerate(neurons):
                        for j, (x, y, _) in enumerate(layer):
                            neurons[i][j] = (x, y, random.uniform(-2, 2))
                    normalization_applied = [False] * len(neurons)
            
            elif reset_button.is_clicked(event.pos):
                if not animating:
                    for i, layer in enumerate(neurons):
                        for j, (x, y, _) in enumerate(layer):
                            neurons[i][j] = (x, y, random.uniform(-2, 2))
                    normalization_applied = [False] * len(neurons)
                    layer_means = [0] * len(neurons)
                    layer_stds = [1] * len(neurons)
    
    # Update button hover states
    normalize_button.update(mouse_pos)
    randomize_button.update(mouse_pos)
    reset_button.update(mouse_pos)
    
    # Handle animation
    if animating:
        animation_progress += 0.02
        if animation_progress >= 1:
            animation_progress = 1
            animating = False
            neurons = animation_target
        else:
            # Interpolate between source and target
            for i in range(len(neurons)):
                for j in range(len(neurons[i])):
                    x, y, _ = neurons[i][j]
                    _, _, src_a = animation_source[i][j]
                    _, _, tgt_a = animation_target[i][j]
                    a = src_a + (tgt_a - src_a) * animation_progress
                    neurons[i][j] = (x, y, a)

    # Drawing
    screen.fill(WHITE)

    # Draw title and subtitle
    title = title_font.render("Batch Normalization in Neural Networks", True, BLACK)
    subtitle = font.render("Developed by : Venugopal Adep", True, GRAY)
    screen.blit(title, (width // 2 - title.get_width() // 2, 20))
    screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 70))

    # Draw layer labels
    for i, name in enumerate(layer_names):
        x = 500 + i * 300
        y = 130
        label = font.render(name, True, BLACK)
        screen.blit(label, (x - label.get_width() // 2, y))

    # Draw connections
    for layer_idx, layer_connections in enumerate(connections):
        for start, end, weight in layer_connections:
            # Calculate color based on weight
            if weight > 0:
                intensity = min(255, int(200 * abs(weight)))
                color = (0, 0, intensity)
            else:
                intensity = min(255, int(200 * abs(weight)))
                color = (intensity, 0, 0)
            
            # Draw line with alpha based on weight strength
            pygame.draw.line(screen, color, start[:2], end[:2], max(1, int(abs(weight) * 3)))

    # Draw neurons
    for layer_idx, layer in enumerate(neurons):
        for x, y, activation in layer:
            # Color gradient from red (negative) to blue (positive)
            normalized_activation = max(-1, min(1, activation / 2))  # Scale to [-1, 1]
            if normalized_activation < 0:
                color = (int(255 * abs(normalized_activation)), 0, 0)
            else:
                color = (0, 0, int(255 * normalized_activation))
            
            # Draw neuron circle
            pygame.draw.circle(screen, color, (int(x), int(y)), 20)
            pygame.draw.circle(screen, BLACK, (int(x), int(y)), 20, 2)
            
            # Display activation value
            value_text = small_font.render(f"{activation:.2f}", True, BLACK)
            screen.blit(value_text, (int(x) - value_text.get_width() // 2, int(y) + 25))

    # Draw information panel
    info_box_x = 50
    info_box_y = 150
    info_box_width = 250
    info_box_height = 300
    
    pygame.draw.rect(screen, LIGHT_GRAY, (info_box_x, info_box_y, info_box_width, info_box_height), border_radius=10)
    pygame.draw.rect(screen, BLACK, (info_box_x, info_box_y, info_box_width, info_box_height), 2, border_radius=10)
    
    info_title = font.render("Information", True, BLACK)
    screen.blit(info_title, (info_box_x + 10, info_box_y + 10))
    
    info_texts = [
        "Batch Normalization:",
        "- Normalizes layer outputs",
        "- Reduces internal covariate shift",
        "- Speeds up training",
        "- Improves gradient flow",
        "",
        "Color Legend:",
        "- Red: Negative values",
        "- Blue: Positive values",
        "- Line thickness: Weight strength"
    ]
    
    for i, text in enumerate(info_texts):
        info_text = small_font.render(text, True, BLACK)
        screen.blit(info_text, (info_box_x + 15, info_box_y + 50 + i * 22))

    # Draw statistics boxes for each layer in a row
    for i in range(1, len(neurons)):  # Skip input layer
        activations = [neuron[2] for neuron in neurons[i]]
        mean = np.mean(activations)
        std = np.std(activations)
        
        # Position stats boxes in a row at the bottom
        stats_x = 350 + (i - 1) * 260
        stats_y = 570
        stats_width = 240
        stats_height = 120
        
        draw_stats_box(stats_x, stats_y, stats_width, stats_height, i, mean, std, normalization_applied[i])
        
        # Draw histograms below stats boxes
        hist_x = stats_x
        hist_y = stats_y + stats_height + 10
        hist_width = stats_width
        hist_height = 120
        
        hist_title = "Activation Distribution"
        draw_histogram(hist_x, hist_y, hist_width, hist_height, activations, hist_title)

    # Draw buttons
    normalize_button.draw()
    randomize_button.draw()
    reset_button.draw()

    # Display normalization formula
    formula_text = "Normalization: z = (x - μ) / σ"
    formula = font.render(formula_text, True, PURPLE)
    screen.blit(formula, (width - 350, 800))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
