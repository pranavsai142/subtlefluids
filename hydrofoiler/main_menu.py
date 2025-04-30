import pygame
import numpy as np

class MainMenu:
    def __init__(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.font = pygame.font.SysFont("arial", 40, bold=True)
        self.buttonFont = pygame.font.SysFont("arial", 30)
        self.title = self.font.render("Hydrofoiler", True, (0, 0, 0))
        self.buttons = [
            {"text": "The Shop", "action": "shop", "rect": None},
            {"text": "Tunnel", "action": "tunnel", "rect": None},
            {"text": "Ocean", "action": "ocean", "rect": None},
            {"text": "Pilot", "action": "pilot", "rect": None}
        ]
        self.updateButtonRects()

    def updateButtonRects(self):
        buttonWidth = 200
        buttonHeight = 50
        spacing = 20
        startY = self.windowHeight // 3
        for i, button in enumerate(self.buttons):
            x = (self.windowWidth - buttonWidth) // 2
            y = startY + i * (buttonHeight + spacing)
            button["rect"] = pygame.Rect(x, y, buttonWidth, buttonHeight)

    def resize(self, size):
        self.windowWidth, self.windowHeight = size
        self.updateButtonRects()

    def handleClick(self, pos, geometrySet):
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                if button["action"] == "shop" or geometrySet:
                    return button["action"]
        return None

    def render(self, surface, geometrySet, geometry=None):
        surface.fill((255, 255, 255))
        titleRect = self.title.get_rect(center=(self.windowWidth // 2, self.windowHeight // 4))
        surface.blit(self.title, titleRect)
        for button in self.buttons:
            if button["action"] in ["tunnel", "ocean", "pilot"] and not geometrySet:
                pygame.draw.rect(surface, (150, 150, 150), button["rect"])
                text = self.buttonFont.render(button["text"], True, (100, 100, 100))
            else:
                pygame.draw.rect(surface, (200, 200, 200), button["rect"])
                text = self.buttonFont.render(button["text"], True, (0, 0, 0))
            textRect = text.get_rect(center=button["rect"].center)
            surface.blit(text, textRect)
        # Render geometry preview if geometry is set
        if geometrySet and geometry:
            previewSize = 100  # Size of the preview square
            previewRect = pygame.Rect(self.windowWidth - previewSize - 10, 10, previewSize, previewSize)
            pygame.draw.rect(surface, (200, 200, 200), previewRect)  # Background
            # Scale geometry to fit within previewSize
            # Access coordinates through geometry.geometry
            xRange = max(geometry.geometry.pointXCoords) - min(geometry.geometry.pointXCoords)
            zRange = max(geometry.geometry.pointZCoords) - min(geometry.geometry.pointZCoords)
            maxRange = max(xRange, zRange, 1e-6)
            scale = (previewSize - 10) / maxRange  # Leave 5px padding on each side
            centroidX = np.mean(geometry.geometry.pointXCoords)
            centroidZ = np.mean(geometry.geometry.pointZCoords)
            points = []
            for x, z in zip(geometry.geometry.pointXCoords, geometry.geometry.pointZCoords):
                screenX = previewRect.left + 5 + (x - centroidX) * scale + previewSize / 2
                screenZ = previewRect.top + 5 + (z - centroidZ) * scale + previewSize / 2
                points.append((screenX, screenZ))
            # Draw the geometry
            for i in range(len(points)):
                start = points[i]
                end = points[(i + 1) % len(points)]
                pygame.draw.line(surface, (0, 0, 255), start, end, 1)

    def _checkGeometrySet(self):
        # This method will be called by render; we need to access Hydrofoiler's geometry
        # Since MainMenu doesn't have direct access, we'll rely on the geometrySet parameter passed during handleClick
        # For rendering, assume geometrySet is passed externally or maintain state if needed
        # For now, we'll adjust the render call in hydrofoiler.py to pass geometrySet
        return True  # Placeholder; actual check happens in hydrofoiler.py