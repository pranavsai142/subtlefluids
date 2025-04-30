import pygame
import numpy as np

class MainMenu:
    def __init__(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.font = pygame.font.SysFont("arial", 40, bold=True)
        self.buttonFont = pygame.font.SysFont("arial", 30)
        self.smallFont = pygame.font.SysFont("arial", 20)
        self.title = self.font.render("Hydrofoiler", True, (0, 0, 0))
        self.buttons = [
            {"text": "The Shop", "action": "shop", "rect": None},
            {"text": "Tunnel", "action": "tunnel", "rect": None},
            {"text": "Ocean", "action": "ocean", "rect": None},
            {"text": "Pilot", "action": "pilot", "rect": None}
        ]
        self.showHelp = False
        self.buttonWidth = 200
        self.buttonHeight = 50
        self.padding = 10
        self.helpButtonWidth = self.buttonWidth // 2
        self.helpButtonHeight = self.buttonHeight // 2
        self.rotationAngle = 0.0  # Track rotation angle in degrees
        self.rotationSpeed = 2.0  # Degrees per frame (adjust for speed)
        self.updateButtonRects()

    def updateButtonRects(self):
        buttonWidth = self.buttonWidth
        buttonHeight = self.buttonHeight
        spacing = 20
        startY = self.windowHeight // 3
        for i, button in enumerate(self.buttons):
            x = (self.windowWidth - buttonWidth) // 2
            y = startY + i * (buttonHeight + spacing)
            button["rect"] = pygame.Rect(x, y, buttonWidth, buttonHeight)
        self.helpButtonRect = pygame.Rect(
            self.windowWidth - self.helpButtonWidth - self.padding,
            self.windowHeight - self.helpButtonHeight - self.padding,
            self.helpButtonWidth,
            self.helpButtonHeight
        )

    def resize(self, size):
        self.windowWidth, self.windowHeight = size
        self.updateButtonRects()

    def handleClick(self, pos, geometrySet):
        if self.helpButtonRect.collidepoint(pos):
            self.showHelp = not self.showHelp
            return None
        if self.showHelp:
            self.showHelp = False
            return None
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                if button["action"] == "shop" or geometrySet:
                    return button["action"]
        return None

    def update(self):
        # Update the rotation angle for the spinning geometry
        self.rotationAngle += self.rotationSpeed
        if self.rotationAngle >= 360:
            self.rotationAngle -= 360  # Keep angle in [0, 360)

    def render(self, surface, geometrySet, geometry=None):
        surface.fill((255, 255, 255))
        titleRect = self.title.get_rect(center=(self.windowWidth // 2, self.windowHeight // 4))
        surface.blit(self.title, titleRect)

        mousePos = pygame.mouse.get_pos()
        for button in self.buttons:
            if button["action"] in ["tunnel", "ocean", "pilot"] and not geometrySet:
                pygame.draw.rect(surface, (150, 150, 150), button["rect"])
                text = self.buttonFont.render(button["text"], True, (100, 100, 100))
            else:
                color = (200, 200, 200) if button["rect"].collidepoint(mousePos) else (150, 150, 150)
                pygame.draw.rect(surface, color, button["rect"])
                text = self.buttonFont.render(button["text"], True, (0, 0, 0))
            textRect = text.get_rect(center=button["rect"].center)
            surface.blit(text, textRect)

        helpColor = (200, 200, 200) if self.helpButtonRect.collidepoint(mousePos) else (150, 150, 150)
        pygame.draw.rect(surface, helpColor, self.helpButtonRect)
        helpText = self.smallFont.render("Help", True, (0, 0, 0))
        helpTextRect = helpText.get_rect(center=self.helpButtonRect.center)
        surface.blit(helpText, helpTextRect)

        # Render geometry preview with rotation
        if geometrySet and geometry:
            previewSize = 100
            previewRect = pygame.Rect(self.windowWidth - previewSize - 10, 10, previewSize, previewSize)
            xRange = max(geometry.geometry.pointXCoords) - min(geometry.geometry.pointXCoords)
            zRange = max(geometry.geometry.pointZCoords) - min(geometry.geometry.pointZCoords)
            maxRange = max(xRange, zRange, 1e-6)
            scale = (previewSize - 10) / maxRange
            centroidX = np.mean(geometry.geometry.pointXCoords)
            centroidZ = np.mean(geometry.geometry.pointZCoords)

            # Rotate points around centroid
            angleRad = np.radians(self.rotationAngle)
            cosA = np.cos(angleRad)
            sinA = np.sin(angleRad)
            points = []
            for x, z in zip(geometry.geometry.pointXCoords, geometry.geometry.pointZCoords):
                # Translate to origin (relative to centroid)
                xRel = x - centroidX
                zRel = z - centroidZ
                # Rotate
                xRot = xRel * cosA - zRel * sinA
                zRot = xRel * sinA + zRel * cosA
                # Translate back and scale to screen coordinates
                screenX = previewRect.left + 5 + (xRot + maxRange / 2) * scale
                screenZ = previewRect.top + 5 + (zRot + maxRange / 2) * scale
                points.append((screenX, screenZ))

            # Draw the rotated geometry
            for i in range(len(points)):
                start = points[i]
                end = points[(i + 1) % len(points)]
                pygame.draw.line(surface, (0, 0, 255), start, end, 1)

        if self.showHelp:
            helpOverlay = pygame.Surface((self.windowWidth, self.windowHeight), pygame.SRCALPHA)
            helpOverlay.fill((200, 200, 200, 200))
            controls = [
                "Rotate Left: Left Arrow",
                "Rotate Right: Right Arrow",
                "Zoom In: Mouse Wheel Up",
                "Zoom Out: Mouse Wheel Down",
                "Thrust Forward (Pilot): Up Arrow",
                "Thrust Backward (Pilot): Down Arrow",
                "Translate Frame: WASD",
                "Reset View: H",
                "Quit to Main Menu: Escape",
                "Open Menu: U"
            ]
            yStart = (self.windowHeight - len(controls) * 30) // 2
            for i, control in enumerate(controls):
                text = self.smallFont.render(control, True, (0, 0, 0))
                textRect = text.get_rect(center=(self.windowWidth // 2, yStart + i * 30))
                helpOverlay.blit(text, textRect)
            surface.blit(helpOverlay, (0, 0))

    def _checkGeometrySet(self):
        return True