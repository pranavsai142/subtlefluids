import pygame
import numpy as np
import math

class UIManager:
    def __init__(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.digitHeight = 20
        self.digitFont = pygame.font.SysFont("courier", 16, bold=True)
        self.labelFont = pygame.font.SysFont("ocraextended", 40, bold=False)
        self.textColor = (0, 0, 0)
        self.digitBgColor = (50, 50, 50)
        self.digitColor = (255, 255, 255)
        self.digitWindowColor = (30, 30, 30)
        self.orientBgColor = (200, 200, 200, 128)
        self.hiddenUIShown = False
        self.tunnelParams = {
            "velocity": 10.0,  # Maximum velocity
            "alpha": 0.0,     # Angle of attack
            "unsteady": False,
            "T": 100.0        # Period of oscillation
        }
        self.oceanParams = {
            "deltaX": 100.0,
            "deltaZ": 100.0
        }
        self.uiElements = []
        self.environment = None
        self._updateDimensions(windowWidth, windowHeight)
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]
        self.signWheel = self._createSignWheel()
        self.wheelOffsets = {}
        self.animationSpeed = 15
        self.animationDuration = 10
        self.animationFrames = {}
        self.prevValues = {}
        self.prevAoa = {}
        self.smoothingFactor = 0.1
        self.frameNumber = 0
        self.screen = pygame.display.get_surface()

    def _createUIElements(self):
        elements = []
        if self.hiddenUIShown:
            y = self.uiPanelRect.y + 10
            spacing = 40
            if self.environment == "tunnel":
                # Unsteady checkbox
                unsteadyRect = pygame.Rect(self.uiPanelRect.x + 10, y, 25, 25)
                elements.append({"type": "checkbox", "label": "Unst:", "value": self.tunnelParams["unsteady"], "rect": unsteadyRect, "key": "unsteady"})
                y += spacing
                # Base fields: velocity and alpha
                fields = [
                    ("Vel:", str(self.tunnelParams["velocity"]), "velocity"),
                    ("α:", str(self.tunnelParams["alpha"]), "alpha"),
                ]
                # Add period (T) field only if unsteady is True
                if self.tunnelParams["unsteady"]:
                    unsteadyFields = [
                        ("T:", str(self.tunnelParams["T"]), "T"),
                    ]
                    fields.extend(unsteadyFields)
                for label, value, key in fields:
                    rect = pygame.Rect(self.uiPanelRect.x + 10, y, 150, 25)
                    elements.append({"type": "textbox", "label": label, "value": value, "rect": rect, "key": key, "active": False})
                    y += spacing
            elif self.environment == "ocean" or self.environment == "pilot":
                fields = [
                    ("ΔX:", str(self.oceanParams["deltaX"]), "deltaX"),
                    ("ΔZ:", str(self.oceanParams["deltaZ"]), "deltaZ")
                ]
                for label, value, key in fields:
                    rect = pygame.Rect(self.uiPanelRect.x + 10, y, 150, 25)
                    elements.append({"type": "textbox", "label": label, "value": value, "rect": rect, "key": key, "active": False})
                    y += spacing
        return elements

    def _updateDimensions(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.readoutWidth = windowWidth // 6
        self.readoutHeight = windowHeight
        self.readoutPos = (windowWidth - self.readoutWidth, 0)
        self.digitWidth = (self.readoutWidth - 60) // 4
        self.uiPanelRect = pygame.Rect(50, 50, 300, 400)
        self.uiElements = self._createUIElements()

    def resize(self, size):
        self._updateDimensions(*size)
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]
        self.signWheel = self._createSignWheel()

    def _createDigitWheel(self):
        wheelHeight = self.digitHeight * 10
        wheel = pygame.Surface((self.digitWidth, wheelHeight), pygame.SRCALPHA)
        for i in range(10):
            digitText = self.digitFont.render(str(i), True, self.digitColor)
            xPos = (self.digitWidth - digitText.get_width()) // 2
            yPos = i * self.digitHeight + (self.digitHeight - digitText.get_height()) // 2
            wheel.blit(digitText, (xPos, yPos))
        return wheel

    def _createSignWheel(self):
        wheelHeight = self.digitHeight * 2
        wheel = pygame.Surface((self.digitWidth, wheelHeight), pygame.SRCALPHA)
        for i, symbol in enumerate(["+", "-"]):
            symbolText = self.digitFont.render(symbol, True, self.digitColor)
            xPos = (self.digitWidth - symbolText.get_width()) // 2
            yPos = i * self.digitHeight + (self.digitHeight - symbolText.get_height()) // 2
            wheel.blit(symbolText, (xPos, yPos))
        return wheel

    def toggleHiddenUI(self):
        self.hiddenUIShown = not self.hiddenUIShown
        self.uiElements = self._createUIElements()

    def handleClick(self, pos):
        x, y = pos
        for element in self.uiElements:
            if element["rect"].collidepoint(x, y):
                if element["type"] == "textbox":
                    element["active"] = True
                elif element["type"] == "checkbox":
                    if element["key"] == "unsteady":
                        self.tunnelParams["unsteady"] = not self.tunnelParams["unsteady"]
                        element["value"] = self.tunnelParams["unsteady"]
                        self.uiElements = self._createUIElements()
                for other in self.uiElements:
                    if other != element and other["type"] == "textbox":
                        other["active"] = False
        return False
    
    def handleKey(self, event):
        for element in self.uiElements:
            if element.get("active", False) and element["type"] == "textbox":
                if event.key == pygame.K_BACKSPACE:
                    element["value"] = element["value"][:-1]
                elif event.key == pygame.K_RETURN:
                    element["active"] = False
                    try:
                        value = float(element["value"])
                        if self.environment == "tunnel":
                            self.tunnelParams[element["key"]] = value
                        elif self.environment in ["ocean", "pilot"]:
                            self.oceanParams[element["key"]] = value
                    except ValueError:
                        element["value"] = str(self.tunnelParams.get(element["key"], 
                                                                     self.oceanParams.get(element["key"], 0.0)))
                elif event.unicode.isprintable():
                    element["value"] += event.unicode

    def getTunnelFlowParams(self):
        velocity = float(self.tunnelParams["velocity"])
        alphaDeg = float(self.tunnelParams["alpha"])
        unsteady = self.tunnelParams["unsteady"]
        T = float(self.tunnelParams["T"])
        velocity = 10.0
        alphaDeg = 0.0
        unsteady = True
        T = 100
        return velocity, alphaDeg, unsteady, T

    def getOceanParams(self):
        return self.oceanParams["deltaX"], self.oceanParams["deltaZ"]

    def renderShopUI(self, shop):
        pass

    def renderEnvironmentUI(self, environmentObj, envType):
        self.environment = envType
        self.frameNumber = environmentObj.frameNumber
        self._renderReadoutView(environmentObj, self.screen)
        if self.hiddenUIShown:
            pygame.draw.rect(self.screen, self.orientBgColor, self.uiPanelRect, border_radius=5)
            smallFont = pygame.font.SysFont("ocraextended", 20, bold=False)
            for element in self.uiElements:
                if element["type"] == "textbox":
                    if element.get("active", False):
                        pygame.draw.rect(self.screen, (255, 165, 0), element["rect"], 2, border_radius=3)
                    pygame.draw.rect(self.screen, (220, 220, 220), element["rect"], border_radius=3)
                    label = smallFont.render(element["label"], True, self.textColor)
                    self.screen.blit(label, (element["rect"].x - 50, element["rect"].y + 5))
                    value = smallFont.render(element["value"], True, self.textColor)
                    self.screen.blit(value, (element["rect"].x + 5, element["rect"].y + 5))
                elif element["type"] == "checkbox":
                    pygame.draw.rect(self.screen, (220, 220, 220), element["rect"], border_radius=3)
                    if element["value"]:
                        pygame.draw.line(self.screen, (0, 0, 0), 
                                         (element["rect"].x + 5, element["rect"].y + 12), 
                                         (element["rect"].x + 12, element["rect"].y + 20), 2)
                        pygame.draw.line(self.screen, (0, 0, 0), 
                                         (element["rect"].x + 12, element["rect"].y + 20), 
                                         (element["rect"].x + 20, element["rect"].y + 5), 2)
                    label = smallFont.render(element["label"], True, self.textColor)
                    self.screen.blit(label, (element["rect"].x - 50, element["rect"].y + 5))

    def _renderReadoutView(self, environmentObj, surface):
        overlay = pygame.Surface((self.readoutWidth, self.readoutHeight), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        yPos = self.readoutPos[1] + 10
        if self.environment == "ocean":
            labels = ["POS X:", "POS Z:", "VEL X:", "VEL Z:", "ROT ANG:", "AOA:"]
        elif self.environment == "pilot":
            labels = ["POS X:", "POS Z:", "VEL X:", "VEL Z:", "THRUST:", "ROT ANG:", "AOA:"]
        else:
            labels = ["VEL X:", "VEL Z:", "ACC X:", "ACC Z:", "ROT ANG:", "AOA:"]
        spacing = (self.readoutHeight - 20) // len(labels)
        for obj in environmentObj.objects:
            if self.environment == "ocean":
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                values = [posX, posZ, velX, velZ]
            elif self.environment == "pilot":
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                thrust = np.linalg.norm(obj.thrustForce)
                values = [posX, posZ, velX, velZ, thrust]
            else:
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                accX, accZ = obj.accelerationVector[0], obj.accelerationVector[1]
                velX, velZ = -velX, -velZ
                accX, accZ = -accX, -accZ
                accX = accX * 10
                accZ = accZ * 10
                values = [velX, velZ, accX, accZ]
            orient = np.array(obj.orientationVector, dtype=float)
            refVector = np.array([0, -1], dtype=float)
            normOrient = np.linalg.norm(orient)
            normRef = np.linalg.norm(refVector)
            if normOrient > 0 and normRef > 0:
                cosTheta = np.dot(orient, refVector) / (normOrient * normRef)
                cosTheta = np.clip(cosTheta, -1.0, 1.0)
                rotAngle = math.degrees(math.acos(cosTheta))
                cross = orient[0] * refVector[1] - orient[1] * refVector[0]
                if cross < 0:
                    rotAngle = -rotAngle
                rotAngle = ((rotAngle + 180) % 360) - 180
            else:
                rotAngle = 0.0
            vel = np.array(obj.velocityVector, dtype=float)
            normVel = np.linalg.norm(vel)
            normOrient = np.linalg.norm(orient)
            if normOrient > 0 and normVel > 0:
                orientNorm = orient / normOrient
                velNorm = vel / normVel
                cosTheta = np.dot(orientNorm, velNorm)
                cosTheta = np.clip(cosTheta, -1.0, 1.0)
                aoa = math.degrees(math.acos(cosTheta))
                cross = orientNorm[0] * velNorm[1] - orientNorm[1] * velNorm[0]
                if cross < 0:
                    aoa = -aoa
                aoa = ((aoa + 180) % 360) - 180
                key = f"{id(obj)}_AoA"
                if key in self.prevAoa:
                    aoa = self.prevAoa[key] + self.smoothingFactor * (aoa - self.prevAoa[key])
                self.prevAoa[key] = aoa
            else:
                aoa = 0.0
            values.extend([rotAngle, aoa])
            for idx, (label, value) in enumerate(zip(labels, values)):
                key = f"{id(obj)}_{label}_value"
                if key not in self.prevValues:
                    self.prevValues[key] = value
                smoothedValue = self.prevValues[key] + self.smoothingFactor * (value - self.prevValues[key])
                self.prevValues[key] = smoothedValue
                value = smoothedValue
                numDigits = 3
                if label in ["ROT ANG:", "AOA:"]:
                    maxValue = 180
                    minValue = -180
                    value = max(min(value, maxValue), minValue)
                    valueStr = f"{abs(int(value)):03d}"
                else:
                    maxValue = 999
                    minValue = -999
                    value = max(min(value, maxValue), minValue)
                    absValue = abs(value)
                    hundreds = int(absValue // 100) % 10
                    tens = int(absValue // 10) % 10
                    ones = int(absValue) % 10
                    valueStr = f"{hundreds:01d}{tens:01d}{ones:01d}"
                signIdx = 0 if value >= 0 else 1
                totalWheels = 4
                key = f"{id(obj)}_{label}"
                if key not in self.wheelOffsets:
                    self.wheelOffsets[key] = [0] * totalWheels
                if key not in self.animationFrames:
                    self.animationFrames[key] = [0] * totalWheels
                labelText = self.labelFont.render(label, True, self.textColor)
                labelX = 5
                labelY = yPos + 2 - self.readoutPos[1]
                overlay.blit(labelText, (labelX, labelY))
                windowY = yPos + 50 - self.readoutPos[1]
                windowX = 5
                windowWidth = self.digitWidth * totalWheels
                pygame.draw.rect(overlay, self.digitWindowColor, (windowX, windowY, windowWidth, self.digitHeight))
                if self.environment == "tunnel" and label in ["ACC X:", "ACC Z:"]:
                    decimalX = windowX + self.digitWidth * 3
                    decimalY = windowY + self.digitHeight - 3
                    pygame.draw.circle(overlay, self.digitColor, (decimalX, decimalY), 1)
                wheelsToDraw = [(self.signWheel, signIdx, 2)]
                for i, digit in enumerate(valueStr):
                    wheelsToDraw.append((self.digitWheels[i], int(digit), 10))
                for i, (wheel, targetDigit, numPositions) in enumerate(wheelsToDraw):
                    xPos = windowX + i * self.digitWidth
                    currentOffset = self.wheelOffsets[key][i]
                    targetOffset = targetDigit * self.digitHeight
                    diff = targetOffset - currentOffset
                    totalHeight = self.digitHeight * numPositions
                    if diff > totalHeight / 2:
                        diff -= totalHeight
                    elif diff < -totalHeight / 2:
                        diff += totalHeight
                    if diff != 0:
                        self.animationFrames[key][i] += 1
                        step = self.animationSpeed if diff > 0 else -self.animationSpeed
                        currentOffset += step
                        if abs(currentOffset - targetOffset) < self.animationSpeed or self.animationFrames[key][i] >= self.animationDuration:
                            currentOffset = targetOffset
                            self.animationFrames[key][i] = 0
                        currentOffset = currentOffset % totalHeight
                        if currentOffset < 0:
                            currentOffset += totalHeight
                    else:
                        self.animationFrames[key][i] = 0
                    self.wheelOffsets[key][i] = currentOffset
                    wheelPos = windowY - currentOffset
                    clipRect = pygame.Rect(xPos, windowY, self.digitWidth, self.digitHeight)
                    overlay.set_clip(clipRect)
                    overlay.blit(wheel, (xPos, wheelPos))
                    overlay.blit(wheel, (xPos, wheelPos - totalHeight))
                    overlay.blit(wheel, (xPos, wheelPos + totalHeight))
                    overlay.set_clip(None)
                pygame.draw.rect(overlay, self.textColor, (windowX, windowY, windowWidth, self.digitHeight), 1)
                yPos += spacing
        background = pygame.Surface((self.readoutWidth, self.readoutHeight), pygame.SRCALPHA)
        background.fill(self.orientBgColor)
        background.blit(overlay, (0, 0))
        surface.blit(background, self.readoutPos)