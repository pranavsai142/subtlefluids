import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hydrofoil Drop Simulator")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
DARK_BLUE = (0, 50, 100)
GRAY = (150, 150, 150)

# Font
font = pygame.font.Font(None, 36)

# Game states
MENU, DESIGN, SIMULATION, CUTSCENE = 0, 1, 2, 3
game_state = MENU

# Constants
GRAVITY = 9.81  # m/s^2
MASS = 1.0      # kg (simplified)
DT = 1/60       # Time step (at 60 FPS)

# Hydrofoil class
class Hydrofoil:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = 50
        self.angle = -90  # Nose down (orientation angle)
        self.velocity_x = 0
        self.velocity_y = 0
        self.lift = 0
        self.drag = 0
        self.shape = "NACA0012"
        self.angle_of_attack = 0
        self.force_x = 0
        self.force_y = 0
        
    def update(self):
        # Calculate all forces
        self.calculate_forces()
        
        # Add gravity force (downward)
        self.force_y += MASS * GRAVITY
        
        # Calculate acceleration (F = ma)
        accel_x = self.force_x / MASS
        accel_y = self.force_y / MASS
        
        # Update velocities
        self.velocity_x += accel_x * DT
        self.velocity_y += accel_y * DT
        
        # Update position
        self.x += self.velocity_x * DT
        self.y += self.velocity_y * DT
        
        # Reset forces for next frame
        self.force_x = 0
        self.force_y = 0
        
        # Keep within screen bounds
        self.x = max(0, min(self.x, WIDTH))
        if self.y > HEIGHT:
            self.y = HEIGHT
            self.velocity_y = 0
            self.velocity_x *= 0.9  # Friction
            
#     TODO: Calculate lift and drag using simplified assumptions accuratley
    def calculate_forces(self):
        # Calculate apparent current (velocity relative to water)
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        velocity_angle = 0
        if speed > 0:
            velocity_angle = math.degrees(math.atan2(self.velocity_y, self.velocity_x))
            self.angle_of_attack = self.angle - velocity_angle
            self.angle_of_attack = ((self.angle_of_attack + 180) % 360)
            if(self.angle_of_attack > 180):
                self.angle_of_attack = self.angle_of_attack - 360
        else:
            self.angle_of_attack = 0
            
        # Convert to radians for calculations
        aoa_rad = math.radians(self.angle_of_attack)
        
        # Lift and drag coefficients
#         cl = 2 * math.pi * math.sin(aoa_rad)
#         cd = 0.1 + abs(math.sin(aoa_rad))
        cl = 0
        cd = 0
        # Apparent flow magnitude
        apparent_flow = speed
        
        # Calculate forces
        self.lift = cl * apparent_flow**2 * 0.5
        self.drag = cd * apparent_flow**2 * 0.5
        
        # Apply forces in global coordinates
        flow_angle = math.radians(velocity_angle)
        # Lift is perpendicular to flow
        lift_x = -self.lift * math.sin(flow_angle)
        lift_y = self.lift * math.cos(flow_angle)
        # Drag is opposite to flow direction
        drag_x = -self.drag * math.cos(flow_angle)
        drag_y = -self.drag * math.sin(flow_angle)
        
        # Add forces to total
        self.force_x = lift_x + drag_x
        self.force_y = lift_y + drag_y
        
    def draw(self, surface):
        points = [
            (self.x + 20 * math.cos(math.radians(self.angle)), 
             self.y + 20 * math.sin(math.radians(self.angle))),
            (self.x + 60 * math.cos(math.radians(self.angle + 10)), 
             self.y + 60 * math.sin(math.radians(self.angle + 10))),
            (self.x - 40 * math.cos(math.radians(self.angle)), 
             self.y - 40 * math.sin(math.radians(self.angle))),
            (self.x + 60 * math.cos(math.radians(self.angle - 10)), 
             self.y + 60 * math.sin(math.radians(self.angle - 10)))
        ]
        pygame.draw.polygon(surface, GRAY, points)

# UI elements
class Button:
    def __init__(self, text, x, y, width, height):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        
    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, self.rect)
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def clicked(self, pos):
        return self.rect.collidepoint(pos)

# TODO: Select geometry from a menu
# TODO: Draw custom geometry
# Game objects
foil = Hydrofoil()
buttons = [
    Button("Design", 300, 200, 200, 50),
    Button("Start Simulation", 300, 300, 200, 50),
    Button("Back to Menu", 300, 400, 200, 50)
]

# Ocean background
def draw_ocean():
    for y in range(0, HEIGHT, 20):
        blue_value = max(50, min(255, 255 - y//3))
        pygame.draw.rect(screen, (0, 50, blue_value), (0, y, WIDTH, 20))

# Main game loop
clock = pygame.time.Clock()
running = True
cutscene_timer = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if game_state == MENU:
                if buttons[0].clicked(event.pos):
                    game_state = DESIGN
                elif buttons[1].clicked(event.pos):
                    game_state = CUTSCENE
                    cutscene_timer = 120
            elif game_state == DESIGN and buttons[2].clicked(event.pos):
                game_state = MENU
        elif event.type == pygame.KEYDOWN and game_state == SIMULATION:
            if event.key == pygame.K_LEFT:
                foil.angle += 5
            elif event.key == pygame.K_RIGHT:
                foil.angle -= 5

    # Draw background
    draw_ocean()

    if game_state == MENU:
        for button in buttons[:2]:
            button.draw(screen)
            
    elif game_state == DESIGN:
        buttons[2].draw(screen)
        text = font.render(f"Selected Foil: {foil.shape}", True, WHITE)
        screen.blit(text, (50, 50))
        
    elif game_state == CUTSCENE:
        text = font.render("Dropping Hydrofoil...", True, WHITE)
        screen.blit(text, (WIDTH//2-100, HEIGHT//2))
        cutscene_timer -= 1
        if cutscene_timer <= 0:
            game_state = SIMULATION
            foil = Hydrofoil()
            
    elif game_state == SIMULATION:
        foil.update()
        foil.draw(screen)
        
        # Draw UI overlay
        ui_texts = [
            f"Vert Velocity: {foil.velocity_y:.2f} m/s",
            f"Horz Velocity: {foil.velocity_x:.2f} m/s",
            f"X Force: {foil.force_x:.2f} N",
            f"Y Force: {foil.force_y:.2f} N",
            f"Lift: {foil.lift:.2f} N",
            f"Drag: {foil.drag:.2f} N",
            f"Angle: {foil.angle:.1f}°",
            f"AoA: {foil.angle_of_attack:.1f}°"
        ]
        pygame.draw.rect(screen, DARK_BLUE, (WIDTH-200, 0, 200, 250))
        for i, text in enumerate(ui_texts):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (WIDTH-190, 10 + i*25))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()   