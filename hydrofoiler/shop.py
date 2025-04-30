import pygame
import numpy as np
from geometry import Circle, Foil

class Shop:
    def __init__(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.mode = "select"  # select, draw, set_params
        self.geometryType = None  # circle, foil
        self.points = []  # For manual foil drawing
        self.drawingRect = pygame.Rect(100, 100, 400, 300)
        self.symmetryLineY = self.drawingRect.bottom
        self.nacaPreset = "0012"
        self.chordLength = 0.203
        self.mass = 300
        self.font = pygame.font.SysFont("arial", 20)
        self.buttons = [
            {"text": "Circle", "action": "circle", "rect": pygame.Rect(50, 50, 100, 40)},
            {"text": "Foil", "action": "foil", "rect": pygame.Rect(160, 50, 100, 40)},
            {"text": "NACA 0012", "action": "naca", "rect": None},
            {"text": "Manual Draw", "action": "manual", "rect": None},
            {"text": "Set Geometry", "action": "set_geometry", "rect": None},
            {"text": "Back", "action": "back", "rect": pygame.Rect(50, windowHeight - 50, 100, 40)},
            {"text": "Confirm Points", "action": "confirm_points", "rect": None}  # New button
        ]
        self.inputFields = [
            {"label": "Chord Length:", "value": str(self.chordLength), "rect": None, "active": False},
            {"label": "Mass:", "value": str(self.mass), "rect": None, "active": False}
        ]
        self.updateUIRects()

    def updateUIRects(self):
        buttonWidth = 150
        buttonHeight = 40
        spacing = 10
        startY = self.windowHeight - 150
        for i, button in enumerate(self.buttons[2:-2], 2):
            button["rect"] = pygame.Rect(50, startY + (i-2) * (buttonHeight + spacing), buttonWidth, buttonHeight)
        self.buttons[-1]["rect"] = pygame.Rect(self.drawingRect.right + 10, self.drawingRect.bottom - 40, 150, 40)  # Confirm Points button
        labelWidth = 120
        for i, field in enumerate(self.inputFields):
            field["rect"] = pygame.Rect(50 + labelWidth + 10, startY + i * (buttonHeight + spacing), 150, buttonHeight)

    def resize(self, size):
        self.windowWidth, self.windowHeight = size
        self.buttons[-1]["rect"] = pygame.Rect(50, self.windowHeight - 50, 100, 40)
        self.updateUIRects()

    def handleClick(self, pos):
        for button in self.buttons:
            if button["rect"] and button["rect"].collidepoint(pos):
                if button["action"] == "circle":
                    self.geometryType = "circle"
                    self.mode = "set_params"
                elif button["action"] == "foil":
                    self.geometryType = "foil"
                    self.mode = "select_foil"
                elif button["action"] == "naca":
                    self.mode = "set_params"
                elif button["action"] == "manual":
                    self.mode = "draw"
                    self.points = []
                elif button["action"] == "set_geometry":
                    return self.createGeometry()
                elif button["action"] == "back":
                    return "menu"
                elif button["action"] == "confirm_points":
                    if self.points:  # Only proceed if points exist
                        self.mode = "set_params"
        for field in self.inputFields:
            if field["rect"].collidepoint(pos):
                field["active"] = True
            else:
                field["active"] = False
        if self.mode == "draw" and self.drawingRect.collidepoint(pos):
            x, z = pos[0] - self.drawingRect.x, self.drawingRect.height - (pos[1] - self.drawingRect.y)
            if x >= 0 and z >= 0:  # Only add points in the upper half
                self.points.append((x / self.drawingRect.width, z / self.drawingRect.height))
        return None

    def handleKey(self, event):
        for field in self.inputFields:
            if field["active"]:
                if event.key == pygame.K_BACKSPACE:
                    field["value"] = field["value"][:-1]
                elif event.key == pygame.K_RETURN:
                    field["active"] = False
                elif event.unicode.isdigit() or event.unicode == ".":
                    field["value"] += event.unicode
        # Removed the ENTER key handling for manual draw

    def createGeometry(self):
        try:
            self.chordLength = float(self.inputFields[0]["value"])
            self.mass = float(self.inputFields[1]["value"])
        except ValueError:
            return None
        if self.geometryType == "circle":
            geometry = Circle(self.chordLength, self.mass)
        else:
            if self.mode == "set_params" and self.points:
                # Convert points to nacaXRaw, nacaZRaw, adding nose and tail points at Z=0
                xRaw = [p[0] for p in self.points]
                zRaw = [p[1] for p in self.points]
                # Normalize X coordinates to [0, 1]
                minX = min(xRaw)
                maxX = max(xRaw)
                if maxX == minX:  # Avoid division by zero
                    xRawNormalized = [0.0] * len(xRaw)
                else:
                    xRawNormalized = [(x - minX) / (maxX - minX) for x in xRaw]
                # Add nose and tail points at Z=0, using normalized min and max X (0 and 1)
                points = [(x, z) for x, z in zip(xRawNormalized, zRaw) if x != 0.0 and x != 1.0]
                xRawNormalized, zRaw = zip(*points) if points else ([], [])
                xRawNormalized = list(xRawNormalized)
                zRaw = list(zRaw)
                xRawNormalized.insert(0, 0.0)
                zRaw.insert(0, 0)
                xRawNormalized.append(1.0)
                zRaw.append(0)
                # Sort points by x to ensure proper ordering
                points = sorted(zip(xRawNormalized, zRaw), key=lambda p: p[0])
                xRawNormalized, zRaw = zip(*points)
                # Apply chord length scaling
                xRaw = [x * self.chordLength for x in xRawNormalized]
                zRaw = [z * self.chordLength for z in zRaw]
                geometry = Foil(points=(xRaw, zRaw), chordLength=self.chordLength, numPoints=300, mass=self.mass)
            else:
                geometry = Foil(self.nacaPreset, self.chordLength, numPoints=300, mass=self.mass)
        return geometry

    def render(self, surface):
        surface.fill((255, 255, 255))
        if self.mode == "select":
            for button in self.buttons[:2] + [self.buttons[-2]]:
                pygame.draw.rect(surface, (200, 200, 200), button["rect"])
                text = self.font.render(button["text"], True, (0, 0, 0))
                textRect = text.get_rect(center=button["rect"].center)
                surface.blit(text, textRect)
        elif self.mode == "select_foil":
            for button in self.buttons[2:4] + [self.buttons[-2]]:
                pygame.draw.rect(surface, (200, 200, 200), button["rect"])
                text = self.font.render(button["text"], True, (0, 0, 0))
                textRect = text.get_rect(center=button["rect"].center)
                surface.blit(text, textRect)
        elif self.mode == "draw":
            pygame.draw.rect(surface, (220, 220, 220), self.drawingRect)
            pygame.draw.line(surface, (0, 0, 0), (self.drawingRect.left, self.symmetryLineY),
                             (self.drawingRect.right, self.symmetryLineY), 2)
            for x, z in self.points:
                px = self.drawingRect.x + x * self.drawingRect.width
                py = self.drawingRect.y + (1 - z) * self.drawingRect.height
                pygame.draw.circle(surface, (0, 0, 255), (px, py), 5)
                mirrorPy = self.symmetryLineY + (self.symmetryLineY - py)
                pygame.draw.circle(surface, (0, 0, 255), (px, mirrorPy), 5)
            confirmButton = self.buttons[-1]
            pygame.draw.rect(surface, (200, 200, 200), confirmButton["rect"])
            text = self.font.render(confirmButton["text"], True, (0, 0, 0))
            textRect = text.get_rect(center=confirmButton["rect"].center)
            surface.blit(text, textRect)
        elif self.mode == "set_params":
            for field in self.inputFields:
                # Highlight active text box
                if field.get("active", False):
                    pygame.draw.rect(surface, (255, 165, 0), field["rect"], 2)  # Orange border
                pygame.draw.rect(surface, (200, 200, 200), field["rect"])
                labelText = "Radius:" if self.geometryType == "circle" and field["label"] == "Chord Length:" else field["label"]
                label = self.font.render(labelText, True, (0, 0, 0))
                labelX = field["rect"].x - 150
                surface.blit(label, (labelX, field["rect"].y + 5))
                value = self.font.render(field["value"], True, (0, 0, 0))
                surface.blit(value, (field["rect"].x + 5, field["rect"].y + 5))
            setButton = self.buttons[4]
            pygame.draw.rect(surface, (200, 200, 200), setButton["rect"])
            text = self.font.render(setButton["text"], True, (0, 0, 0))
            textRect = text.get_rect(center=setButton["rect"].center)
            surface.blit(text, textRect)