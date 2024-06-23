import pygame
import math
import random
import time

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Propagation Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Neuron parameters
num_inputs = 3
weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
bias = random.uniform(-1, 1)
learning_rate = 0.1

# Font
font = pygame.font.Font(None, 24)
arrow_font = pygame.font.Font(None, 36)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def neuron_output(inputs):
    weighted_sum = sum(w * i for w, i in zip(weights, inputs)) + bias
    return sigmoid(weighted_sum)

def draw_neuron(x, y, radius, color):
    pygame.draw.circle(screen, color, (x, y), radius)

def draw_inputs(inputs):
    for i, input_val in enumerate(inputs):
        x = 200
        y = 250 + i * 150
        draw_neuron(x, y, 30, RED)
        text = font.render(f"Input {i+1}: {input_val:.2f}", True, BLACK)
        screen.blit(text, (x + 40, y - 10))

def draw_weights():
    for i, weight in enumerate(weights):
        x1, y1 = 200, 250 + i * 150
        x2, y2 = 600, 400
        pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 2)
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        text = font.render(f"W{i+1}: {weight:.2f}", True, BLACK)
        screen.blit(text, (mid_x, mid_y))

def draw_output(output):
    x, y = 1000, 400
    draw_neuron(x, y, 30, GREEN)
    text = font.render(f"Output: {output:.4f}", True, BLACK)
    screen.blit(text, (x + 40, y - 10))

def draw_error(error):
    text = font.render(f"Error: {error:.4f}", True, BLACK)
    screen.blit(text, (900, 500))

def draw_target(target):
    text = font.render(f"Target: {target:.4f}", True, BLACK)
    screen.blit(text, (900, 550))

def draw_epoch(epoch):
    text = font.render(f"Epoch: {epoch}", True, BLACK)
    screen.blit(text, (900, 600))

def draw_arrow(start, end, color, text):
    pygame.draw.line(screen, color, start, end, 3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_size = 20
    pygame.draw.polygon(screen, color, [
        (end[0] - arrow_size * math.cos(angle - math.pi/6),
         end[1] - arrow_size * math.sin(angle - math.pi/6)),
        end,
        (end[0] - arrow_size * math.cos(angle + math.pi/6),
         end[1] - arrow_size * math.sin(angle + math.pi/6))
    ])
    text_surface = arrow_font.render(text, True, color)
    text_rect = text_surface.get_rect(center=((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 20))
    screen.blit(text_surface, text_rect)

def draw_button(x, y, width, height, text, color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

def draw_slider(x, y, width, height, value, min_val, max_val):
    pygame.draw.rect(screen, GRAY, (x, y, width, height))
    slider_pos = int(x + (value - min_val) / (max_val - min_val) * width)
    pygame.draw.rect(screen, BLUE, (slider_pos - 5, y - 5, 10, height + 10))
    text = font.render(f"Learning Rate: {value:.3f}", True, BLACK)
    screen.blit(text, (x, y - 30))

def main():
    global weights, bias, learning_rate
    inputs = [random.random() for _ in range(num_inputs)]
    target = random.random()
    output = neuron_output(inputs)
    error = target - output
    running = True
    clock = pygame.time.Clock()
    forward_prop = False
    backward_prop = False
    flash_timer = 0
    epoch = 0
    forward_completed = False
    auto_mode = False
    last_auto_time = 0
    slider_dragging = False

    while running:
        current_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    x, y = event.pos
                    if 10 <= x <= 210 and HEIGHT - 60 <= y <= HEIGHT - 10:
                        auto_mode = not auto_mode
                        last_auto_time = current_time
                    elif HEIGHT - 110 <= y <= HEIGHT - 90:
                        slider_dragging = True
                    else:
                        for i in range(num_inputs):
                            input_x, input_y = 200, 250 + i * 150
                            if math.hypot(x - input_x, y - input_y) < 30:
                                inputs[i] = random.random()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    slider_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if slider_dragging:
                    x, _ = event.pos
                    learning_rate = max(0.001, min(1.0, (x - 250) / 300))

        screen.fill(WHITE)

        # Draw neuron components
        draw_neuron(600, 400, 40, BLUE)
        draw_inputs(inputs)
        draw_weights()
        draw_output(output)
        draw_error(error)
        draw_target(target)
        draw_epoch(epoch)

        # Draw bias
        bias_text = font.render(f"Bias: {bias:.4f}", True, BLACK)
        screen.blit(bias_text, (550, 350))

        # Draw auto mode button
        button_color = GREEN if auto_mode else GRAY
        draw_button(10, HEIGHT - 60, 200, 50, "Auto Mode: " + ("ON" if auto_mode else "OFF"), button_color)

        # Draw learning rate slider
        draw_slider(250, HEIGHT - 110, 300, 20, learning_rate, 0.001, 1.0)

        # Instructions
        instructions = [
            "Click on input circles to randomize their values",
            "Press 'R' to reset weights and bias",
            "Press 'F' for forward propagation",
            "Press 'B' for backward propagation",
            "Press 'T' for new random target",
        ]
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, BLACK)
            screen.blit(text, (10, 10 + i * 30))

        # Auto mode logic
        if auto_mode:
            if not forward_prop and not backward_prop and current_time - last_auto_time >= 0.5:
                if not forward_completed:
                    forward_prop = True
                else:
                    backward_prop = True
                last_auto_time = current_time

        # Forward propagation animation
        if forward_prop:
            if flash_timer < 30:
                draw_arrow((200, 400), (1000, 400), BLUE, "FORWARD")
                flash_timer += 1
            else:
                output = neuron_output(inputs)
                error = target - output
                forward_prop = False
                flash_timer = 0
                forward_completed = True

        # Backward propagation animation
        if backward_prop:
            if flash_timer < 30:
                draw_arrow((1000, 400), (200, 400), RED, "BACKWARD")
                flash_timer += 1
            else:
                output_delta = error * sigmoid_derivative(output)
                for i in range(num_inputs):
                    weight_delta = output_delta * inputs[i]
                    weights[i] += learning_rate * weight_delta
                bias += learning_rate * output_delta
                backward_prop = False
                flash_timer = 0
                if forward_completed:
                    epoch += 1
                    forward_completed = False

        pygame.display.flip()
        clock.tick(60)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
            bias = random.uniform(-1, 1)
            epoch = 0
            forward_completed = False
        elif keys[pygame.K_f]:
            forward_prop = True
        elif keys[pygame.K_b]:
            backward_prop = True
        elif keys[pygame.K_t]:
            target = random.random()

    pygame.quit()

if __name__ == "__main__":
    main()