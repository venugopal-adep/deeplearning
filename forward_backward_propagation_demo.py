import pygame
import math
import random
import time

# Initialize Pygame
pygame.init()

# Set up the display with 1600x900 resolution
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Neural Network Propagation Demo")

# Enhanced color palette
BACKGROUND = (240, 240, 245)
BLACK = (40, 40, 40)
RED = (220, 80, 80)
GREEN = (80, 200, 100)
BLUE = (60, 120, 210)
YELLOW = (240, 200, 60)
GRAY = (180, 180, 190)
PURPLE = (150, 100, 200)
TEAL = (80, 200, 180)

# Neuron parameters
num_inputs = 3
weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
bias = random.uniform(-1, 1)
learning_rate = 0.1

# Improved fonts
title_font = pygame.font.Font(None, 48)
font = pygame.font.Font(None, 28)
arrow_font = pygame.font.Font(None, 36)

# Grid parameters for 3D effect
grid_size = 30
grid_color = (230, 230, 240)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def neuron_output(inputs):
    weighted_sum = sum(w * i for w, i in zip(weights, inputs)) + bias
    return sigmoid(weighted_sum)

def draw_3d_neuron(x, y, radius, color, glow=False, z_offset=0):
    # Add shadow for 3D effect
    shadow_offset = radius // 3
    shadow = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    pygame.draw.circle(shadow, (0, 0, 0, 80), (radius, radius), radius)
    screen.blit(shadow, (x - radius + shadow_offset + z_offset, y - radius + shadow_offset))
    
    if glow:
        # Add glow effect
        for i in range(3):
            glow_radius = radius + (3-i)*5
            glow_surface = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            alpha = 100 - i*30
            pygame.draw.circle(glow_surface, (*color[:3], alpha), (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surface, (x-glow_radius, y-glow_radius))
    
    # Draw neuron with gradient for 3D effect
    for i in range(radius, 0, -2):
        shade = tuple(min(255, c + (radius - i) * 2) for c in color[:3])
        pygame.draw.circle(screen, shade, (x, y), i)
    
    # Add highlight for 3D effect
    highlight = pygame.Surface((radius//2, radius//2), pygame.SRCALPHA)
    pygame.draw.circle(highlight, (255, 255, 255, 100), (radius//4, radius//4), radius//4)
    screen.blit(highlight, (x - radius//3, y - radius//3))

def draw_button(x, y, width, height, text, color, hover=False):
    # Draw button with gradient and shadow
    if hover:
        shadow_offset = 3
        shadow = pygame.Surface((width, height))
        shadow.fill((50, 50, 50))
        shadow.set_alpha(100)
        screen.blit(shadow, (x + shadow_offset, y + shadow_offset))
        
        # Lighter color when hovering
        color = tuple(min(255, c + 30) for c in color)
    
    # Button gradient
    for i in range(height):
        gradient_color = tuple(max(0, c - i//2) for c in color)
        pygame.draw.line(screen, gradient_color, (x, y + i), (x + width, y + i))
    
    # Button border
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    # Button text without shadow
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

def draw_slider(x, y, width, height, value, min_val, max_val):
    # Draw slider track with gradient
    for i in range(width):
        gradient_color = (
            int(GRAY[0] * (1 - i/width) + BLUE[0] * (i/width)),
            int(GRAY[1] * (1 - i/width) + BLUE[1] * (i/width)),
            int(GRAY[2] * (1 - i/width) + BLUE[2] * (i/width))
        )
        pygame.draw.line(screen, gradient_color, (x + i, y), (x + i, y + height))
    
    # Draw slider border
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 1)
    
    # Draw slider handle with glow effect
    slider_pos = int(x + (value - min_val) / (max_val - min_val) * width)
    
    # Glow effect
    for i in range(3):
        glow_size = 15 - i*3
        glow_surface = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        alpha = 100 - i*30
        pygame.draw.circle(glow_surface, (*BLUE[:3], alpha), (glow_size, glow_size), glow_size)
        screen.blit(glow_surface, (slider_pos - glow_size, y + height//2 - glow_size))
    
    # Slider handle
    pygame.draw.circle(screen, BLUE, (slider_pos, y + height//2), 10)
    pygame.draw.circle(screen, BLACK, (slider_pos, y + height//2), 10, 1)
    
    # Slider label with value
    text = font.render(f"Learning Rate: {value:.3f}", True, BLACK)
    screen.blit(text, (x, y - 30))



def draw_inputs(inputs):
    for i, input_val in enumerate(inputs):
        x = WIDTH // 5
        y = HEIGHT // 4 + i * (HEIGHT // 4)
        z_offset = 10  # 3D depth effect
        draw_3d_neuron(x, y, 40, RED, z_offset=z_offset)
        
        # Add 3D text effect with shadow
        text = font.render(f"Input {i+1}: {input_val:.2f}", True, BLACK)
        shadow_text = font.render(f"Input {i+1}: {input_val:.2f}", True, (100, 100, 100))
        screen.blit(shadow_text, (x + 52, y - 8))
        screen.blit(text, (x + 50, y - 10))

def draw_3d_line(start, end, color, thickness=3):
    # Draw shadow line for 3D effect
    shadow_offset = 5
    pygame.draw.line(screen, (0, 0, 0, 100), 
                    (start[0] + shadow_offset, start[1] + shadow_offset), 
                    (end[0] + shadow_offset, end[1] + shadow_offset), 
                    thickness)
    pygame.draw.line(screen, color, start, end, thickness)

def draw_weights():
    for i, weight in enumerate(weights):
        x1, y1 = WIDTH // 5, HEIGHT // 4 + i * (HEIGHT // 4)
        x2, y2 = WIDTH // 2, HEIGHT // 2
        
        # Calculate line color based on weight value
        if weight > 0:
            line_color = (0, min(255, int(200 * weight)), 0)
        else:
            line_color = (min(255, int(-200 * weight)), 0, 0)
            
        # Draw line with thickness based on absolute weight value
        thickness = max(1, min(8, int(abs(weight) * 5)))
        draw_3d_line((x1, y1), (x2, y2), line_color, thickness)
        
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Add 3D text effect with shadow and background
        text_bg = pygame.Surface((100, 30), pygame.SRCALPHA)
        text_bg.fill((255, 255, 255, 180))
        screen.blit(text_bg, (mid_x - 5, mid_y - 3))
        
        weight_text = font.render(f"W{i+1}: {weight:.2f}", True, BLACK)
        shadow_text = font.render(f"W{i+1}: {weight:.2f}", True, (100, 100, 100))
        screen.blit(shadow_text, (mid_x + 2, mid_y + 2))
        screen.blit(weight_text, (mid_x, mid_y))

def draw_output(output):
    x, y = WIDTH * 4 // 5, HEIGHT // 2
    draw_3d_neuron(x, y, 40, GREEN, glow=True)
    
    # Add 3D text effect with shadow
    text = font.render(f"Output: {output:.4f}", True, BLACK)
    screen.blit(text, (x + 50, y - 10))

def draw_central_neuron():
    x, y = WIDTH // 2, HEIGHT // 2
    draw_3d_neuron(x, y, 50, BLUE, glow=True)
    
    # Draw bias connection with 3D effect
    bias_x, bias_y = x - 80, y - 80
    draw_3d_line((bias_x, bias_y), (x, y), PURPLE)
    
    # Draw bias neuron with 3D effect
    draw_3d_neuron(bias_x, bias_y, 25, PURPLE)
    
    # Draw bias label with 3D effect
    bias_text = font.render(f"Bias: {bias:.4f}", True, BLACK)
    shadow_text = font.render(f"Bias: {bias:.4f}", True, (100, 100, 100))
    
    bias_bg = pygame.Surface((bias_text.get_width() + 10, bias_text.get_height() + 6), pygame.SRCALPHA)
    bias_bg.fill((255, 255, 255, 180))
    screen.blit(bias_bg, (bias_x - bias_text.get_width()//2, bias_y - 45))
    
    screen.blit(shadow_text, (bias_x - bias_text.get_width()//2 + 7, bias_y - 38))
    screen.blit(bias_text, (bias_x - bias_text.get_width()//2 + 5, bias_y - 40))

def draw_3d_arrow(start, end, color, text, active=False):
    # Draw shadow for 3D effect
    shadow_offset = 5
    
    # Draw arrow shaft with 3D effect
    if active:
        # Animated arrow with particles
        num_segments = 20
        for i in range(num_segments):
            segment_start = (
                start[0] + (end[0] - start[0]) * i / num_segments,
                start[1] + (end[1] - start[1]) * i / num_segments
            )
            segment_end = (
                start[0] + (end[0] - start[0]) * (i + 1) / num_segments,
                start[1] + (end[1] - start[1]) * (i + 1) / num_segments
            )
            segment_color = (
                color[0],
                color[1],
                color[2],
                int(255 * (i + 1) / num_segments)
            )
            
            # Shadow for 3D effect
            shadow_segment = (
                segment_start[0] + shadow_offset,
                segment_start[1] + shadow_offset
            )
            shadow_end = (
                segment_end[0] + shadow_offset,
                segment_end[1] + shadow_offset
            )
            pygame.draw.line(screen, (*BLACK, 100), shadow_segment, shadow_end, 4)
            
            pygame.draw.line(screen, segment_color, segment_start, segment_end, 4)
            
            # Add particles for 3D effect
            if random.random() < 0.3:
                particle_pos = (
                    segment_start[0] + random.randint(-10, 10),
                    segment_start[1] + random.randint(-10, 10)
                )
                particle_size = random.randint(2, 5)
                particle_color = (color[0], color[1], color[2], 150)
                particle_surface = pygame.Surface((particle_size*2, particle_size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, particle_color, (particle_size, particle_size), particle_size)
                screen.blit(particle_surface, (particle_pos[0]-particle_size, particle_pos[1]-particle_size))
    else:
        # Draw shadow for 3D effect
        pygame.draw.line(screen, (*BLACK, 100), 
                        (start[0] + shadow_offset, start[1] + shadow_offset), 
                        (end[0] + shadow_offset, end[1] + shadow_offset), 3)
        pygame.draw.line(screen, color, start, end, 3)
    
    # Draw arrowhead with 3D effect
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_size = 20
    
    # Shadow for arrowhead
    shadow_end = (end[0] + shadow_offset, end[1] + shadow_offset)
    pygame.draw.polygon(screen, (*BLACK, 100), [
        (shadow_end[0] - arrow_size * math.cos(angle - math.pi/6),
         shadow_end[1] - arrow_size * math.sin(angle - math.pi/6)),
        shadow_end,
        (shadow_end[0] - arrow_size * math.cos(angle + math.pi/6),
         shadow_end[1] - arrow_size * math.sin(angle + math.pi/6))
    ])
    
    # Actual arrowhead
    pygame.draw.polygon(screen, color, [
        (end[0] - arrow_size * math.cos(angle - math.pi/6),
         end[1] - arrow_size * math.sin(angle - math.pi/6)),
        end,
        (end[0] - arrow_size * math.cos(angle + math.pi/6),
         end[1] - arrow_size * math.sin(angle + math.pi/6))
    ])
    
    # Draw text without 3D effect
    text_surface = arrow_font.render(text, True, color)
    text_bg = pygame.Surface((text_surface.get_width() + 10, text_surface.get_height() + 6), pygame.SRCALPHA)
    text_bg.fill((255, 255, 255, 180))
    text_rect = text_bg.get_rect(center=((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 20))
    screen.blit(text_bg, text_rect)
    text_rect = text_surface.get_rect(center=((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 20))
    screen.blit(text_surface, text_rect)

def draw_info_panel(error, target, epoch):
    panel_width = 300
    panel_height = 150
    panel_x = WIDTH - panel_width - 20
    panel_y = HEIGHT - panel_height - 20
    
    # Draw panel with 3D effect
    shadow = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 80))
    screen.blit(shadow, (panel_x + 5, panel_y + 5))
    
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 200))
    screen.blit(panel, (panel_x, panel_y))
    
    # Draw border with 3D effect
    pygame.draw.rect(screen, BLACK, (panel_x, panel_y, panel_width, panel_height), 2)
    
    # Draw info text with 3D effect
    info_items = [
        ("Error:", f"{error:.4f}"),
        ("Target:", f"{target:.4f}"),
        ("Epoch:", f"{epoch}"),
        ("Learning Rate:", f"{learning_rate:.3f}")
    ]
    
    for i, (label, value) in enumerate(info_items):
        label_text = font.render(label, True, BLACK)
        value_text = font.render(value, True, BLUE)
        
        y_pos = panel_y + 20 + i * 30
        screen.blit(label_text, (panel_x + 20, y_pos))
        screen.blit(value_text, (panel_x + 150, y_pos))

def draw_title():
    title = title_font.render("3D Neural Network Propagation Demo", True, BLACK)
    
    # Add background panel without 3D effect
    title_bg = pygame.Surface((title.get_width() + 40, title.get_height() + 20), pygame.SRCALPHA)
    title_bg.fill((255, 255, 255, 180))
    
    # Remove shadow background and only keep the main background
    screen.blit(title_bg, (WIDTH//2 - title.get_width()//2 - 20, 20))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

def draw_instructions():
    instructions = [
        "R: Reset weights and bias",
        "F: Forward propagation",
        "B: Backward propagation",
        "T: New random target",
        "Click inputs to randomize"
    ]
    
    panel_width = 300
    panel_height = 30 * len(instructions) + 20
    panel_x = 20
    panel_y = 20
    
    # Draw panel with 3D effect
    shadow = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 80))
    screen.blit(shadow, (panel_x + 5, panel_y + 5))
    
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 200))
    screen.blit(panel, (panel_x, panel_y))
    
    # Draw border with 3D effect
    pygame.draw.rect(screen, BLACK, (panel_x, panel_y, panel_width, panel_height), 2)
    
    for i, instruction in enumerate(instructions):
        text = font.render(instruction, True, BLACK)
        screen.blit(text, (panel_x + 20, panel_y + 20 + i * 30))

def draw_3d_grid():
    # Draw a 3D grid effect in the background
    for x in range(0, WIDTH, grid_size):
        intensity = 255 - int(150 * (x / WIDTH))
        color = (intensity, intensity, intensity)
        pygame.draw.line(screen, color, (x, 0), (x, HEIGHT))
    
    for y in range(0, HEIGHT, grid_size):
        intensity = 255 - int(150 * (y / HEIGHT))
        color = (intensity, intensity, intensity)
        pygame.draw.line(screen, color, (0, y), (WIDTH, y))

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
    
    # Particle system for visual effects
    particles = []
    
    while running:
        current_time = time.time()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Check button hover
        auto_button_hover = 10 <= mouse_x <= 210 and HEIGHT - 60 <= mouse_y <= HEIGHT - 10
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    x, y = event.pos
                    if 10 <= x <= 210 and HEIGHT - 60 <= y <= HEIGHT - 10:
                        auto_mode = not auto_mode
                        last_auto_time = current_time
                    elif HEIGHT - 110 <= y <= HEIGHT - 90 and 250 <= x <= 550:
                        slider_dragging = True
                    else:
                        for i in range(num_inputs):
                            input_x, input_y = WIDTH // 5, HEIGHT // 4 + i * (HEIGHT // 4)
                            if math.hypot(x - input_x, y - input_y) < 40:
                                inputs[i] = random.random()
                                # Add particles for visual feedback
                                for _ in range(20):
                                    particles.append({
                                        'x': input_x,
                                        'y': input_y,
                                        'vx': random.uniform(-2, 2),
                                        'vy': random.uniform(-2, 2),
                                        'color': RED,
                                        'size': random.randint(2, 5),
                                        'life': 30
                                    })
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    slider_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if slider_dragging:
                    x, _ = event.pos
                    learning_rate = max(0.001, min(1.0, (x - 250) / 300))
        
        # Draw 3D background with grid
        screen.fill(BACKGROUND)
        draw_3d_grid()
        
        # Draw title
        draw_title()
        
        # Draw neuron components
        draw_central_neuron()
        draw_inputs(inputs)
        draw_weights()
        draw_output(output)
        
        # Draw info panel
        draw_info_panel(error, target, epoch)
        
        # Draw instructions
        draw_instructions()
        
        # Draw auto mode button
        button_color = GREEN if auto_mode else GRAY
        draw_button(10, HEIGHT - 60, 200, 50, "Auto Mode: " + ("ON" if auto_mode else "OFF"), button_color, hover=auto_button_hover)
        
        # Draw learning rate slider
        draw_slider(250, HEIGHT - 110, 300, 20, learning_rate, 0.001, 1.0)
        
        # Update and draw particles
        for particle in particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            if particle['life'] <= 0:
                particles.remove(particle)
            else:
                alpha = min(255, particle['life'] * 8)
                particle_color = (*particle['color'][:3], alpha)
                particle_surface = pygame.Surface((particle['size']*2, particle['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, particle_color, (particle['size'], particle['size']), particle['size'])
                screen.blit(particle_surface, (particle['x']-particle['size'], particle['y']-particle['size']))
        
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
                draw_3d_arrow((WIDTH // 5, HEIGHT // 2), (WIDTH * 4 // 5, HEIGHT // 2), BLUE, "FORWARD", active=True)
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
                draw_3d_arrow((WIDTH * 4 // 5, HEIGHT // 2), (WIDTH // 5, HEIGHT // 2), RED, "BACKWARD", active=True)
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
        
        # Handle keyboard input
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
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()

