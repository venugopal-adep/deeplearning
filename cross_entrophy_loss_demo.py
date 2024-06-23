import pygame
import numpy as np
import math

# Initialize Pygame
pygame.init()
width, height = 1800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Cross Entropy Loss in Neural Networks - Interactive Demo")

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
small_font = pygame.font.Font(None, 24)
formula_font = pygame.font.Font(None, 36)

# Cross Entropy Loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))

# Slider class
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False

    def draw(self):
        pygame.draw.rect(screen, GRAY, self.rect)
        pos = self.rect.x + int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.circle(screen, BLUE, (pos, self.rect.centery), 10)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            self.value = self.min_val + (self.max_val - self.min_val) * (rel_x / self.rect.width)
            self.value = max(self.min_val, min(self.max_val, self.value))

# Create sliders for predicted probabilities
sliders = [
    Slider(400, 200, 600, 20, 0, 1, 0.25),
    Slider(400, 300, 600, 20, 0, 1, 0.25),
    Slider(400, 400, 600, 20, 0, 1, 0.25),
    Slider(400, 500, 600, 20, 0, 1, 0.25)
]

# True labels (one-hot encoded)
y_true = [0, 1, 0, 0]

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for slider in sliders:
            slider.handle_event(event)

    # Drawing
    screen.fill(WHITE)

    # Draw title
    title = title_font.render("Cross Entropy Loss in Neural Networks", True, BLACK)
    screen.blit(title, (width // 2 - title.get_width() // 2, 20))

    # Draw developer credit
    credit = font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(credit, (width // 2 - credit.get_width() // 2, 70))

    # Draw formula
    formula = formula_font.render("Cross Entropy Loss = -Σ y_true * log(y_pred)", True, BLACK)
    screen.blit(formula, (width // 2 - formula.get_width() // 2, 120))

    # Draw sliders and labels
    for i, slider in enumerate(sliders):
        slider.draw()
        label = font.render(f"Class {i+1}: {slider.value:.2f}", True, BLACK)
        screen.blit(label, (1050, 190 + i * 100))
        true_label = font.render(f"True: {y_true[i]}", True, RED)
        screen.blit(true_label, (300, 190 + i * 100))

    # Calculate and normalize predicted probabilities
    y_pred = [slider.value for slider in sliders]
    y_pred = np.array(y_pred) / np.sum(y_pred)

    # Calculate loss
    loss = cross_entropy_loss(y_true, y_pred)

    # Draw loss
    loss_text = font.render(f"Cross Entropy Loss: {loss:.4f}", True, BLACK)
    screen.blit(loss_text, (width // 2 - loss_text.get_width() // 2, 600))

    # Draw explanation
    explanation = [
        "Cross Entropy Loss measures the performance of a classification model where the predicted output is a probability between 0 and 1.",
        "It increases as the predicted probability diverges from the true label.",
        "Lower loss indicates better predictions.",
        "Try adjusting the sliders to see how the loss changes. The correct class is Class 2 (second slider)."
    ]
    for i, line in enumerate(explanation):
        text = small_font.render(line, True, BLACK)
        screen.blit(text, (50, 650 + i * 30))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()