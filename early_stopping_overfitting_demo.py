import pygame
import random
import math

# Initialize Pygame
pygame.init()
width, height = 1400, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Early Stopping in Neural Networks - Interactive Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Fonts
font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 32)

# Generate curve data
def generate_curve(iterations, start, end, noise, curve_type):
    if curve_type == "training":
        return [max(0, min(1, start + (end - start) * math.log(1 + i) / math.log(iterations) + random.uniform(-noise, noise))) for i in range(iterations)]
    else:  # validation
        return [max(0, min(1, start + (end - start) * math.log(1 + i) / math.log(iterations) + 0.1 * math.sin(i / 5) + random.uniform(-noise, noise))) for i in range(iterations)]

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = small_font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
reset_button = Button(1200, 100, 150, 70, "Reset", GREEN)
pause_button = Button(1200, 200, 150, 70, "Pause", BLUE)

# Main variables
iterations = 200
current_iteration = 0
training_data = generate_curve(iterations, 0.8, 0.1, 0.02, "training")
validation_data = generate_curve(iterations, 0.7, 0.3, 0.05, "validation")
early_stopping_point = None
paused = False
overfitting_threshold = 0.1

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if reset_button.is_clicked(event.pos):
                current_iteration = 0
                training_data = generate_curve(iterations, 0.8, 0.1, 0.02, "training")
                validation_data = generate_curve(iterations, 0.7, 0.3, 0.05, "validation")
                early_stopping_point = None
            elif pause_button.is_clicked(event.pos):
                paused = not paused

    if not paused and current_iteration < iterations:
        current_iteration += 1

    # Check for early stopping
    if current_iteration > 10 and early_stopping_point is None:
        if validation_data[current_iteration-1] > validation_data[current_iteration-10] + overfitting_threshold:
            early_stopping_point = current_iteration - 10

    # Drawing
    screen.fill(WHITE)

    # Draw axes
    pygame.draw.line(screen, BLACK, (100, 700), (1100, 700), 3)
    pygame.draw.line(screen, BLACK, (100, 100), (100, 700), 3)

    # Draw grid
    for i in range(1, 11):
        pygame.draw.line(screen, GRAY, (100, 700 - 60*i), (1100, 700 - 60*i), 1)
        pygame.draw.line(screen, GRAY, (100 + 100*i, 100), (100 + 100*i, 700), 1)

    # Draw curves
    for i in range(1, current_iteration):
        pygame.draw.line(screen, BLUE, (100 + 5 * (i-1), 700 - 600 * training_data[i-1]),
                         (100 + 5 * i, 700 - 600 * training_data[i]), 3)
        pygame.draw.line(screen, RED, (100 + 5 * (i-1), 700 - 600 * validation_data[i-1]),
                         (100 + 5 * i, 700 - 600 * validation_data[i]), 3)

    # Draw early stopping line
    if early_stopping_point:
        pygame.draw.line(screen, GREEN, (100 + 5 * early_stopping_point, 100),
                         (100 + 5 * early_stopping_point, 700), 3)

    # Add labels
    screen.blit(font.render("Error", True, BLACK), (20, 20))
    screen.blit(font.render("Iterations", True, BLACK), (550, 720))
    screen.blit(small_font.render("Training", True, BLUE), (1150, 100))
    screen.blit(small_font.render("Validation", True, RED), (1150, 140))

    # Draw buttons
    reset_button.draw()
    pause_button.draw()

    # Display current iteration and early stopping point
    screen.blit(small_font.render(f"Iteration: {current_iteration}", True, BLACK), (1150, 300))
    if early_stopping_point:
        screen.blit(small_font.render(f"Early Stop: {early_stopping_point}", True, GREEN), (1150, 340))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()