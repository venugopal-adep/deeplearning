import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ML vs DL")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 32)
text_font = pygame.font.Font(None, 24)

# ML and DL data
ml_data = []
dl_data = []
ml_line = None
nn_layers = [4, 6, 5, 3]  # Input, hidden, and output layers

# Button
button_rect = pygame.Rect(WIDTH // 2 - 75, HEIGHT - 50, 150, 40)
button_text = text_font.render("Generate Data", True, BLACK)

def generate_data():
    global ml_data, dl_data, ml_line
    ml_data = [(random.randint(50, 750), random.randint(400, 700)) for _ in range(50)]
    dl_data = [(random.randint(850, 1550), random.randint(400, 700)) for _ in range(50)]
    
    # Calculate ML line
    x_sum = sum(x for x, _ in ml_data)
    y_sum = sum(y for _, y in ml_data)
    xy_sum = sum(x * y for x, y in ml_data)
    x_sq_sum = sum(x * x for x, _ in ml_data)
    n = len(ml_data)
    
    m = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
    b = (y_sum - m * x_sum) / n
    
    ml_line = (m, b)

def draw_neural_network(x, y, layer_sizes):
    layer_spacing = 150
    neuron_radius = 15
    max_neurons = max(layer_sizes)
    layer_height = 300

    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            neuron_x = x + i * layer_spacing
            neuron_y = y + j * (layer_height / (layer_size - 1))
            
            pygame.draw.circle(screen, BLUE, (int(neuron_x), int(neuron_y)), neuron_radius)
            
            if i < len(layer_sizes) - 1:
                for k in range(layer_sizes[i + 1]):
                    next_x = x + (i + 1) * layer_spacing
                    next_y = y + k * (layer_height / (layer_sizes[i+1] - 1))
                    pygame.draw.line(screen, GREEN, (neuron_x, neuron_y), (next_x, next_y), 1)

def draw():
    screen.fill(WHITE)
    
    # Draw title and subtitle
    title = title_font.render("ML vs DL", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    subtitle = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 80))
    
    # Draw dividing line
    pygame.draw.line(screen, BLACK, (WIDTH // 2, 100), (WIDTH // 2, HEIGHT - 60), 2)
    
    # Draw ML side
    ml_title = subtitle_font.render("Machine Learning", True, BLACK)
    screen.blit(ml_title, (WIDTH // 4 - ml_title.get_width() // 2, 120))
    
    for x, y in ml_data:
        pygame.draw.circle(screen, RED, (x, y), 3)
    
    if ml_line:
        m, b = ml_line
        pygame.draw.line(screen, BLUE, (0, b), (800, 800 * m + b), 2)
    
    # Draw DL side
    dl_title = subtitle_font.render("Deep Learning", True, BLACK)
    screen.blit(dl_title, (3 * WIDTH // 4 - dl_title.get_width() // 2, 120))
    
    for x, y in dl_data:
        pygame.draw.circle(screen, RED, (x, y), 3)
    
    draw_neural_network(900, 250, nn_layers)
    
    # Draw button
    pygame.draw.rect(screen, YELLOW, button_rect)
    screen.blit(button_text, (button_rect.centerx - button_text.get_width() // 2, button_rect.centery - button_text.get_height() // 2))
    
    # Draw explanations
    ml_explanation = [
        "Machine Learning:",
        "- Uses simple models",
        "- Often linear relationships",
        "- Requires feature engineering",
        "- Works well with structured data"
    ]
    
    dl_explanation = [
        "Deep Learning:",
        "- Uses neural networks",
        "- Can learn complex patterns",
        "- Automatic feature extraction",
        "- Excels at unstructured data"
    ]
    
    for i, line in enumerate(ml_explanation):
        text = text_font.render(line, True, BLACK)
        screen.blit(text, (50, HEIGHT - 180 + i * 30))
    
    for i, line in enumerate(dl_explanation):
        text = text_font.render(line, True, BLACK)
        screen.blit(text, (WIDTH // 2 + 50, HEIGHT - 180 + i * 30))

    pygame.display.flip()

running = True
generate_data()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                generate_data()
    
    draw()

pygame.quit()