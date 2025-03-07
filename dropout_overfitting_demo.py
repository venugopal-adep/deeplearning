import pygame
import random

# Initialize Pygame
pygame.init()
width, height = 1600, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Dropout in Neural Networks - Interactive Visualization")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
BLUE = (41, 128, 185)
RED = (231, 76, 60)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
PURPLE = (142, 68, 173)
DARK_BLUE = (52, 73, 94)
INACTIVE_COLOR = (189, 195, 199)

# Fonts
title_font = pygame.font.SysFont("Arial", 48, bold=True)
subtitle_font = pygame.font.SysFont("Arial", 36)
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 18)

# Neural network structure
layers = [5, 8, 7, 4]  # Number of neurons in each layer
neurons = []
connections = []

# Initialize neurons with better spacing
sidebar_width = 300
layer_spacing = (width - sidebar_width - 200) // (len(layers) - 1)
neuron_vertical_margin = 120

for i, layer_size in enumerate(layers):
    layer = []
    layer_height = (height - 2 * neuron_vertical_margin)
    for j in range(layer_size):
        x = sidebar_width + 100 + i * layer_spacing
        y = neuron_vertical_margin + j * (layer_height / (layer_size - 1)) if layer_size > 1 else height/2
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
dropout_applied = False

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color or tuple(max(0, c-30) for c in color)
        self.is_hovered = False

    def draw(self):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, DARK_BLUE, self.rect, 2, border_radius=8)
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Slider class
class Slider:
    def __init__(self, x, y, width, height, initial_value=0.5):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_width = 20
        self.handle_rect = pygame.Rect(x + (width - self.handle_width) * initial_value, y - 10, self.handle_width, height + 20)
        self.value = initial_value
        self.active = False

    def draw(self):
        # Draw track
        pygame.draw.rect(screen, LIGHT_GRAY, self.rect, border_radius=5)
        
        # Draw filled portion
        filled_rect = pygame.Rect(self.rect.x, self.rect.y, 
                                 self.rect.width * self.value, self.rect.height)
        pygame.draw.rect(screen, BLUE, filled_rect, border_radius=5)
        
        # Draw handle
        pygame.draw.rect(screen, DARK_BLUE, self.handle_rect, border_radius=10)
        pygame.draw.rect(screen, BLACK, self.handle_rect, 2, border_radius=10)

    def update(self, pos):
        if self.active:
            self.value = (pos[0] - self.rect.x) / self.rect.width
            self.value = max(0, min(1, self.value))
            self.handle_rect.x = self.rect.x + (self.rect.width - self.handle_width) * self.value

    def is_clicked(self, pos):
        return self.handle_rect.collidepoint(pos)

# Create buttons and slider with better positioning
apply_dropout_button = Button(width - 300, height - 150, 200, 50, "Apply Dropout", GREEN)
reset_button = Button(width - 300, height - 80, 200, 50, "Reset Network", YELLOW)
dropout_slider = Slider(width - 350, height - 220, 250, 15, initial_value=0.5)

# Animation variables
animation_active = False
animation_frames = 0
animation_duration = 30  # frames
animation_neurons = []

# Function to apply dropout
def apply_dropout():
    global active_neurons, animation_active, animation_frames, animation_neurons, dropout_applied
    
    dropout_rate = dropout_slider.value
    animation_neurons = [layer.copy() for layer in active_neurons]
    
    for i in range(1, len(layers) - 1):  # Skip input and output layers
        layer_size = len(neurons[i])
        num_dropout = int(layer_size * dropout_rate)
        active_neurons[i] = random.sample(range(layer_size), layer_size - num_dropout)
    
    animation_active = True
    animation_frames = 0
    dropout_applied = True

# Function to reset network
def reset_network():
    global active_neurons, animation_active, animation_frames, animation_neurons, dropout_applied
    
    animation_neurons = [layer.copy() for layer in active_neurons]
    active_neurons = [list(range(len(layer))) for layer in neurons]
    
    animation_active = True
    animation_frames = 0
    dropout_applied = False

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if apply_dropout_button.is_clicked(mouse_pos):
                apply_dropout()
            elif reset_button.is_clicked(mouse_pos):
                reset_network()
            elif dropout_slider.is_clicked(mouse_pos):
                dropout_slider.active = True
        elif event.type == pygame.MOUSEBUTTONUP:
            dropout_slider.active = False
        elif event.type == pygame.MOUSEMOTION:
            dropout_slider.update(mouse_pos)
            apply_dropout_button.update(mouse_pos)
            reset_button.update(mouse_pos)

    # Drawing
    screen.fill(WHITE)
    
    # Draw panel
    info_panel = pygame.Rect(0, 0, sidebar_width, height)
    pygame.draw.rect(screen, LIGHT_GRAY, info_panel)
    
    # Draw title and explanation
    title = title_font.render("Dropout in Neural Networks", True, DARK_BLUE)
    screen.blit(title, (20, 30))
    
    explanations = [
        "Dropout is a regularization technique that",
        "prevents overfitting in neural networks.",
        "",
        "How it works:",
        "• Randomly deactivates neurons during training",
        "• Forces the network to learn redundant patterns",
        "• Neurons can't rely on specific connections",
        "",
        "Benefits:",
        "• Reduces overfitting",
        "• Improves generalization",
        "• Acts like ensemble learning",
        "",
        "This demo shows how dropout randomly",
        "deactivates hidden layer neurons based",
        "on the dropout rate you select."
    ]
    
    for i, line in enumerate(explanations):
        text = small_font.render(line, True, BLACK)
        screen.blit(text, (20, 100 + i * 25))
    
    # Draw layer labels with better positioning
    layer_names = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]
    for i, name in enumerate(layer_names):
        x = sidebar_width + 100 + i * layer_spacing
        text = subtitle_font.render(name, True, DARK_BLUE)
        text_rect = text.get_rect(center=(x, 70))
        screen.blit(text, text_rect)

    # Calculate animation progress
    progress = 1.0
    if animation_active:
        progress = min(1.0, animation_frames / animation_duration)
        animation_frames += 1
        if animation_frames >= animation_duration:
            animation_active = False

    # Draw connections first (so they appear behind neurons)
    for i, layer_connections in enumerate(connections):
        for start, end in layer_connections:
            start_layer = i
            end_layer = i + 1
            start_index = neurons[start_layer].index(start)
            end_index = neurons[end_layer].index(end)
            
            # Determine if connection should be active
            start_active = start_index in active_neurons[start_layer]
            end_active = end_index in active_neurons[end_layer]
            
            # For animation, check previous state
            if animation_active:
                start_was_active = start_index in animation_neurons[start_layer]
                end_was_active = end_index in animation_neurons[end_layer]
                
                # Transition between states
                if start_was_active and not start_active:
                    start_active = progress < 0.5
                if end_was_active and not end_active:
                    end_active = progress < 0.5
            
            if start_active and end_active:
                pygame.draw.line(screen, GRAY, start, end, 1)

    # Draw neurons
    neuron_radius = 25
    for i, layer in enumerate(neurons):
        for j, (x, y) in enumerate(layer):
            active = j in active_neurons[i]
            
            # For animation, check previous state
            if animation_active:
                was_active = j in animation_neurons[i]
                if was_active != active:
                    # Transition between states
                    if progress < 0.5:
                        active = was_active
            
            # Choose color based on layer and active state
            if i == 0:  # Input layer
                color = BLUE
            elif i == len(layers) - 1:  # Output layer
                color = GREEN
            else:  # Hidden layers
                color = PURPLE if active else INACTIVE_COLOR
            
            # Draw neuron
            pygame.draw.circle(screen, color, (x, y), neuron_radius)
            pygame.draw.circle(screen, BLACK, (x, y), neuron_radius, 2)
            
            # Add neuron index
            index_text = small_font.render(str(j+1), True, WHITE)
            index_rect = index_text.get_rect(center=(x, y))
            screen.blit(index_text, index_rect)

    # Draw buttons and slider
    apply_dropout_button.draw()
    reset_button.draw()
    
    # Draw slider with label
    dropout_text = font.render(f"Dropout Rate: {dropout_slider.value:.2f}", True, BLACK)
    screen.blit(dropout_text, (width - 350, height - 250))
    dropout_slider.draw()
    
    # Draw status with better positioning
    status_text = "Status: "
    if dropout_applied:
        status_text += f"Dropout applied with rate {dropout_slider.value:.2f}"
    else:
        status_text += "All neurons active"
    
    status = font.render(status_text, True, DARK_BLUE)
    screen.blit(status, (sidebar_width + 50, height - 50))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
