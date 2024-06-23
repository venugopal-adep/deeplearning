import pygame
import numpy as np
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vanishing Gradient Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (200, 200, 200)

# Fonts
FONT_SMALL = pygame.font.Font(None, 24)
FONT_MEDIUM = pygame.font.Font(None, 32)
FONT_LARGE = pygame.font.Font(None, 48)

# Neural network parameters
num_layers = 10
neurons_per_layer = 5
layer_spacing = WIDTH // (num_layers + 1)
neuron_spacing = HEIGHT // (neurons_per_layer + 1)

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Draw a neuron
def draw_neuron(x, y, activation):
    color = (int(activation * 255), int(activation * 255), int(activation * 255))
    pygame.draw.circle(screen, color, (x, y), 20)
    pygame.draw.circle(screen, BLACK, (x, y), 20, 2)

# Draw text
def draw_text(text, font, color, x, y, align="left"):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if align == "right":
        text_rect.right = x
    elif align == "center":
        text_rect.centerx = x
    else:
        text_rect.left = x
    text_rect.top = y
    screen.blit(text_surface, text_rect)

# Initialize network
network = [np.random.rand(neurons_per_layer) for _ in range(num_layers)]
gradients = [np.zeros(neurons_per_layer) for _ in range(num_layers)]

# Main game loop
def main():
    global network, gradients
    clock = pygame.time.Clock()
    selected_neuron = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for layer in range(num_layers):
                    for neuron in range(neurons_per_layer):
                        neuron_x = (layer + 1) * layer_spacing
                        neuron_y = (neuron + 1) * neuron_spacing
                        if math.sqrt((x - neuron_x)**2 + (y - neuron_y)**2) < 20:
                            selected_neuron = (layer, neuron)
                            # Compute gradients
                            gradients = [np.zeros(neurons_per_layer) for _ in range(num_layers)]
                            gradients[layer][neuron] = sigmoid_derivative(network[layer][neuron])
                            for l in range(layer-1, -1, -1):
                                gradients[l] = sigmoid_derivative(network[l]) * np.mean(gradients[l+1])

        screen.fill(WHITE)

        # Draw title
        draw_text("Vanishing Gradient Demo", FONT_LARGE, BLACK, WIDTH // 2, 20, align="center")
        draw_text("Click on a neuron to see how its gradient propagates backward", FONT_MEDIUM, BLACK, WIDTH // 2, 70, align="center")

        # Draw network
        for layer in range(num_layers):
            for neuron in range(neurons_per_layer):
                x = (layer + 1) * layer_spacing
                y = (neuron + 1) * neuron_spacing
                draw_neuron(x, y, network[layer][neuron])
                
                # Draw connections to previous layer
                if layer > 0:
                    for prev_neuron in range(neurons_per_layer):
                        prev_x = layer * layer_spacing
                        prev_y = (prev_neuron + 1) * neuron_spacing
                        pygame.draw.line(screen, GRAY, (prev_x, prev_y), (x, y), 1)

        # Draw gradients
        if selected_neuron:
            layer, neuron = selected_neuron
            for l in range(layer, -1, -1):
                x = (l + 1) * layer_spacing
                y = (neuron + 1) * neuron_spacing
                gradient = gradients[l][neuron]
                color = (int(255 * (1 - gradient)), int(255 * (1 - gradient)), 255)
                pygame.draw.circle(screen, color, (x, y), 25, 5)
                draw_text(f"{gradient:.3f}", FONT_SMALL, BLACK, x, y + 30, align="center")

        # Draw legend
        draw_text("Gradient strength:", FONT_MEDIUM, BLACK, WIDTH - 250, HEIGHT - 100)
        for i in range(5):
            x = WIDTH - 220 + i * 40
            y = HEIGHT - 50
            color = (int(255 * i / 4), int(255 * i / 4), 255)
            pygame.draw.circle(screen, color, (x, y), 15, 5)
            draw_text(f"{i/4:.1f}", FONT_SMALL, BLACK, x, y + 20, align="center")

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()