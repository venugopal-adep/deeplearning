import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ML vs DL")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (220, 240, 255)
LIGHT_GREEN = (220, 255, 240)
PURPLE = (128, 0, 128)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 36)
text_font = pygame.font.Font(None, 28)

# ML and DL data
ml_data = []
dl_data = []
ml_line = None
nn_layers = [4, 6, 5, 3]  # Input, hidden, and output layers
animation_counter = 0

# Button
button_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 60, 200, 40)
button_text = subtitle_font.render("Generate Data", True, BLACK)

def generate_data():
    global ml_data, dl_data, ml_line
    
    # Generate ML data (linear pattern with noise)
    ml_data = []
    for i in range(50):
        x = random.randint(50, 750)
        y = 550 - x * 0.2 + random.randint(-30, 30)
        ml_data.append((x, y))
    
    # Generate DL data (circular pattern)
    dl_data = []
    center_x, center_y = 1200, 500
    for i in range(50):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.randint(100, 200)
        x = center_x + math.cos(angle) * radius + random.randint(-20, 20)
        y = center_y + math.sin(angle) * radius + random.randint(-20, 20)
        dl_data.append((int(x), int(y)))
    
    # Calculate ML line using linear regression
    x_sum = sum(x for x, _ in ml_data)
    y_sum = sum(y for _, y in ml_data)
    xy_sum = sum(x * y for x, y in ml_data)
    x_sq_sum = sum(x * x for x, _ in ml_data)
    n = len(ml_data)
    
    try:
        m = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
        b = (y_sum - m * x_sum) / n
        ml_line = (m, b)
    except ZeroDivisionError:
        ml_line = (0, 0)

def draw_neural_network(x, y, layer_sizes):
    global animation_counter
    
    layer_spacing = 150
    neuron_radius = 15
    layer_height = 300
    
    # Increment animation counter
    animation_counter = (animation_counter + 1) % 60
    
    # Draw connections first (so they appear behind neurons)
    for i, layer_size in enumerate(layer_sizes):
        if i < len(layer_sizes) - 1:
            for j in range(layer_size):
                neuron_x = x + i * layer_spacing
                neuron_y = y + j * (layer_height / (layer_size - 1))
                
                for k in range(layer_sizes[i + 1]):
                    next_x = x + (i + 1) * layer_spacing
                    next_y = y + k * (layer_height / (layer_sizes[i+1] - 1))
                    
                    # Animate connections
                    if (i + j + k + animation_counter // 10) % 3 == 0:
                        connection_color = GREEN
                        width = 2
                    else:
                        connection_color = (200, 200, 200)
                        width = 1
                    
                    pygame.draw.line(screen, connection_color, (neuron_x, neuron_y), (next_x, next_y), width)
    
    # Draw neurons
    for i, layer_size in enumerate(layer_sizes):
        # Add layer labels
        if i == 0:
            label = text_font.render("Input Layer", True, BLACK)
            screen.blit(label, (x - 40, y - 30))
        elif i == len(layer_sizes) - 1:
            label = text_font.render("Output Layer", True, BLACK)
            screen.blit(label, (x + i * layer_spacing - 40, y - 30))
        else:
            label = text_font.render(f"Hidden Layer {i}", True, BLACK)
            screen.blit(label, (x + i * layer_spacing - 50, y - 30))
        
        for j in range(layer_size):
            neuron_x = x + i * layer_spacing
            neuron_y = y + j * (layer_height / (layer_size - 1))
            
            # Animate neurons
            if (i + j + animation_counter // 10) % 3 == 0:
                color = (100, 100, 255)
            else:
                color = BLUE
            
            # Draw neuron
            pygame.draw.circle(screen, color, (int(neuron_x), int(neuron_y)), neuron_radius)
            pygame.draw.circle(screen, BLACK, (int(neuron_x), int(neuron_y)), neuron_radius, 1)

def draw():
    # Fill background
    screen.fill(WHITE)
    
    # Draw background sections
    pygame.draw.rect(screen, LIGHT_BLUE, (0, 100, WIDTH // 2, HEIGHT - 100), 0)
    pygame.draw.rect(screen, LIGHT_GREEN, (WIDTH // 2, 100, WIDTH // 2, HEIGHT - 100), 0)
    
    # Draw title and subtitle
    title = title_font.render("ML vs DL", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    subtitle = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 70))
    
    # Draw dividing line
    pygame.draw.line(screen, BLACK, (WIDTH // 2, 100), (WIDTH // 2, HEIGHT - 80), 2)
    
    # Draw ML side
    ml_title = subtitle_font.render("Machine Learning", True, PURPLE)
    screen.blit(ml_title, (WIDTH // 4 - ml_title.get_width() // 2, 120))
    
    # Draw ML data points
    for x, y in ml_data:
        pygame.draw.circle(screen, RED, (x, y), 4)
    
    # Draw ML regression line
    if ml_line:
        m, b = ml_line
        pygame.draw.line(screen, BLUE, (50, 50 * m + b), (750, 750 * m + b), 3)
    
    # Draw DL side
    dl_title = subtitle_font.render("Deep Learning", True, PURPLE)
    screen.blit(dl_title, (3 * WIDTH // 4 - dl_title.get_width() // 2, 120))
    
    # Draw DL data points
    for x, y in dl_data:
        pygame.draw.circle(screen, RED, (x, y), 4)
    
    # Draw neural network
    draw_neural_network(1000, 250, nn_layers)
    
    # Draw button
    pygame.draw.rect(screen, YELLOW, button_rect)
    pygame.draw.rect(screen, BLACK, button_rect, 2)
    screen.blit(button_text, (button_rect.centerx - button_text.get_width() // 2, 
                             button_rect.centery - button_text.get_height() // 2))
    
    # Draw explanations
    ml_explanation = [
        "Machine Learning:",
        "• Uses simple models",
        "• Often linear relationships",
        "• Requires feature engineering",
        "• Works well with structured data",
        "• More interpretable results",
        "• Example: Linear Regression, Decision Trees"
    ]
    
    dl_explanation = [
        "Deep Learning:",
        "• Uses neural networks",
        "• Can learn complex patterns",
        "• Automatic feature extraction",
        "• Excels at unstructured data (images, text)",
        "• Often a 'black box'",
        "• Example: CNN, RNN, Transformers"
    ]
    
    for i, line in enumerate(ml_explanation):
        text = text_font.render(line, True, BLACK)
        screen.blit(text, (50, HEIGHT - 250 + i * 30))
    
    for i, line in enumerate(dl_explanation):
        text = text_font.render(line, True, BLACK)
        screen.blit(text, (WIDTH // 2 + 50, HEIGHT - 250 + i * 30))

    pygame.display.flip()

# Main game loop
running = True
generate_data()
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                generate_data()
    
    draw()
    clock.tick(30)  # Control animation speed

pygame.quit()
