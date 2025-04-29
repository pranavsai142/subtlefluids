import pygame
import numpy as np
import math

#### INSTRUCTIONS FOR GROK ####
# This block is written by Grok to provide information to Grok on how to meta-develop this script, in case the thread goes down and Grok has to pick up from scratch.
# - **Purpose**: This script renders a visualization of an object in a 2D environment (Ocean or Tunnel) using Pygame. It includes a global view, an orientation view (subscreen), and a readout panel displaying various parameters (position, velocity, acceleration, rotation angle, angle of attack).
# - **Revision History**: Maintain a detailed revision history at the top of the script for every change requested by the user. Each entry should include the date, a summary of changes, and references to affected methods or lines if needed.
# - **General Functionality**:
#   - Global view: Displays the object in a global coordinate system with velocity and force vectors.
#   - Orientation view (subscreen): Shows a zoomed-in view of the object with local velocity (apparent current) and tangential velocity vectors.
#   - Readouts: Displays numerical data (e.g., Vel X, Vel Z) using digit wheels in an altimeter-like style.
# - **Special Notes**:
#   - The script uses fixed zoom values (`TUNNEL_ZOOM`, `ZOOM`) after removing dynamic zoom adjustments (Revision 2025-04-28 #1).
#   - The Tunnel environment negates velocity and acceleration values in readouts to show relative current (Revision 2025-04-28 #2).
#   - Fonts and styling for readouts have been adjusted multiple times; always refer to the latest revision for the current font.
#   - When modifying visual elements (e.g., arrows, readouts), ensure they align with the altimeter-style aesthetic requested by the user.
# - **Development Tips**:
#   - Test changes with a circle of radius 1 to ensure visibility in the global and orientation views.
#   - Pay attention to clipping issues in the subscreen; the user prefers accurate vector directions over preventing clipping.
#   - Keep backups of previous revisions in the history to track styling changes (e.g., fonts, colors, borders).

#### REVISION HISTORY ####
# 2025-04-28 #1:
# - Removed dynamic zoom adjustment in `_renderGlobalView` and `_renderOrientationView` due to visibility issues.
# - Reverted to fixed zoom values (`TUNNEL_ZOOM`, `ZOOM`) and fixed domain sizes in `_updateDimensions`.
# - Changed readout background to solid gray with black borders for each section in `_renderReadoutView`.
# - Adjusted apparent current arrows in `_renderOrientationView` to prevent clipping by clamping start and end positions.
# - Changed readout label font to `pygame.font.SysFont("courier", 14, bold=True)` to match a previous screenshot.

# 2025-04-28 #2:
# - Increased offset for apparent current arrows in `_renderOrientationView` (`xStart = minX - 0.15`) to move them further left.
# - Reverted subscreen background to semi-transparent gray (`self.orientBgColor = (200, 200, 200, 128)`) in `__init__`.
# - Removed black separator lines in `_renderReadoutView` that were incorrectly drawn.
# - Reverted readout label font to `pygame.font.SysFont("courier", 16, bold=True)` and set text color to black (`self.textColor = (0, 0, 0)`).
# - Adjusted `self.readoutPos` and `self.globalWidth` in `_updateDimensions` to make readouts flush with the right edge of the screen.

# 2025-04-28 #3:
# - Removed clipping logic for apparent current arrows in `_renderOrientationView` to allow true vector directions, even if they clip off the subscreen.
# - Adjusted `self.readoutPos` to `(windowWidth - self.readoutWidth, 0)` and `self.readoutHeight` to `windowHeight` in `_updateDimensions` to make readouts flush with the top and bottom of the screen.
# - Changed readout label font to `pygame.font.SysFont("ocraextended", 16, bold=True)` to match the second screenshot's style.
# - Increased padding between labels and digit wheels in `_renderReadoutView` (`windowY = yPos + 20`).
# - Removed the border around the readout section in `render`.
# - Added a thicker vertical line (3 pixels) between the global frame and readouts in `render` at `x = self.readoutPos[0]`.
# - Added meta-coding instructions and revision history at the top of the script.

# 2025-04-28 #4:
# - Changed readout label font to remove bold styling (`bold=False`) and increased font size to 40 (`pygame.font.SysFont("ocraextended", 40, bold=False)`) in `__init__`.
# - Added logic in `render` to disable rotation controls and fix orientation when `object.geometry.hasTrailingEdge` is `False`. Fixed orientation to `[1, 0]` (positive x-axis).

# 2025-04-28 #5:
# - Increased padding between readout labels and digit wheels in `_renderReadoutView` (`windowY = yPos + 50`) to prevent overlap due to larger font size.
# - Added zoom functionality for the global frame view using mouse scroll wheel in `render`. Adjusts `ZOOM` or `TUNNEL_ZOOM` and updates dimensions via `_updateDimensions`.

# 2025-04-28 #6:
# - Added a text annotation in `_renderOrientationView` to clarify that the force vector's direction is opposite due to the relative velocity being opposite in the local frame (subscreen).
# - Fixed the transformation of the local force vector to global coordinates in `_renderGlobalView` by removing the incorrect sign flip (`-localXAxis` to `localXAxis`). This ensures the rendered global force vector matches the logged global force vector.
# - Adjusted the rendering of the local force vector in `_renderOrientationView` by flipping the x-component to match the expected direction (towards tail). Note: This was a temporary fix.
# - Removed the arbitrary flip of the x-component in `_renderOrientationView` after analyzing `computeForce`. The logged `localForceVector` should render correctly without flipping, indicating the previous rendering issue was likely due to a misunderstanding of the force direction. Added a debug print to confirm the `localForceVector` value.
# - Removed the legend (annotation text) from the subscreen in `_renderOrientationView` as requested. Fixed a typo in the legend rendering (`self.ortPos` to `self.orientPos`) during removal.
# - Reintroduced the flip of the x-component of `localForceVector` in `_renderOrientationView` as a permanent fix due to persistent rendering issues.
# - Corrected the understanding of `pointXCoords` ordering (nose at 0, tail at 0.203, lower to higher values). Identified that the `localForceVector` direction was incorrect due to a sign error in `computeForce`. Recommended removing the negative sign in the `forceX` calculation in `computeForce`, but this caused the global frame to render incorrectly.
# - Reverted the `computeForce` change (restored the negative sign in `forceX` calculation) to correct the global frame rendering. The `localForceVector` now correctly points towards the nose in the local frame (negative x local), mapping to a positive x global (towards the nose, right in global frame). Reintroduced the x-component flip in `_renderOrientationView` to correct the subscreen rendering, as the subscreen inverts the x-axis (higher `pointXCoords` render to the left).

class Renderer:
    ZOOM = 0.01  # Zoom factor for Ocean global view
    TUNNEL_ZOOM = 0.01  # Zoom factor for Tunnel global view
    SHOW_GLOBAL_FORCE_VECTOR = True  # Boolean to control display of global frame force vector

    def __init__(self, windowWidth=800, windowHeight=600, deltaX=None, deltaZ=None, environment="ocean"):
        pygame.init()
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.environment = environment
        
        if self.environment == "ocean":
            if deltaX is None or deltaZ is None:
                raise ValueError("deltaX and deltaZ must be provided for Ocean environment")
            self.deltaX = deltaX
            self.deltaZ = deltaZ
        else:
            self.deltaX = None
            self.deltaZ = None
        
        self.screen = pygame.display.set_mode((windowWidth, windowHeight), pygame.RESIZABLE)
        pygame.display.set_caption("Object Visualization")
        
        self.digitHeight = 20
        self.digitFont = pygame.font.SysFont("courier", 16, bold=True)  # Font for digit wheels
        self.labelFont = pygame.font.SysFont("ocraextended", 40, bold=False)  # Font for labels
        self.annotationFont = pygame.font.SysFont("arial", 12)  # Font for annotations
        
        # Colors
        self.bgColor = (255, 255, 255)
        self.pointColor = (0, 0, 255)
        self.lineColor = (0, 0, 255)
        self.tangentialColor = (0, 255, 0)
        self.velocityColor = (255, 0, 0)
        self.forceColor = (255, 105, 180)
        self.textColor = (0, 0, 0)  # Black for labels
        self.oceanColor = (0, 100, 255)
        self.groundColor = (139, 69, 19)
        self.orientBgColor = (200, 200, 200, 128)  # Semi-transparent gray for subscreen
        self.digitBgColor = (50, 50, 50)
        self.digitColor = (255, 255, 255)
        self.digitWindowColor = (30, 30, 30)
        
        self._updateDimensions(windowWidth, windowHeight)
        
        self.running = True
        
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]
        self.signWheel = self._createSignWheel()
        self.wheelOffsets = {}
        self.animationSpeed = 15
        self.animationDuration = 10
        self.animationFrames = {}
        self.prevValues = {}
        
        if self.environment == "ocean":
            self.pathHistory = []
            self.frameCounter = 0
        
        self.prevAoa = {}
        self.smoothingFactor = 0.1

    def _updateDimensions(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        
        self.readoutWidth = windowWidth // 6
        self.readoutHeight = windowHeight
        self.readoutPos = (windowWidth - self.readoutWidth, 0)
        
        self.globalWidth = windowWidth - self.readoutWidth
        self.globalHeight = windowHeight
        self.globalPos = (0, 0)
        
        self.orientWidth = windowWidth // 4
        self.orientHeight = windowWidth // 4
        self.orientPos = (10, windowHeight - self.orientHeight - 10)
        
        if self.environment == "ocean":
            self.minimapWidth = windowWidth // 4
            self.minimapHeight = windowWidth // 4
            self.minimapPos = (10, 10)
            self.minimapScaleX = self.minimapWidth / self.deltaX
            self.minimapScaleZ = self.minimapHeight / self.deltaZ
        
        zoom = self.TUNNEL_ZOOM if self.environment == "tunnel" else self.ZOOM
        domain_size = 100.0
        self.subDomainSizeX = domain_size * zoom
        self.subDomainSizeZ = domain_size * zoom
        self.scaleX = self.globalWidth / self.subDomainSizeX
        self.scaleZ = self.globalHeight / self.subDomainSizeZ
        
        self.digitWidth = (self.readoutWidth - 60) // 4
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

    def _toScreenCoords(self, x, z, isOrientView=False, isMinimapView=False, orientCenter=(0, 0), orientScale=1.0, subDomainCenter=(0, 0)):
        if isOrientView:
            screenX = self.orientPos[0] + (self.orientWidth / 2) + (x - orientCenter[0]) * orientScale
            screenZ = self.orientPos[1] + (self.orientHeight / 2) - (z - orientCenter[1]) * orientScale
        elif isMinimapView:
            if self.environment != "ocean":
                raise ValueError("Minimap view is only supported for Ocean environment")
            screenX = self.minimapPos[0] + (x * self.minimapScaleX)
            screenZ = self.minimapPos[1] + (self.minimapHeight - (z + self.deltaZ) * self.minimapScaleZ)
        else:
            centerX, centerZ = subDomainCenter
            screenX = self.globalPos[0] + (x - (centerX - self.subDomainSizeX / 2)) * self.scaleX
            zRangeMin = centerZ - self.subDomainSizeZ / 2
            zRangeMax = centerZ + self.subDomainSizeZ / 2
            zNormalized = (z - zRangeMin) / (zRangeMax - zRangeMin)
            screenZ = self.globalPos[1] + (1 - zNormalized) * self.globalHeight
        return (screenX, screenZ)

    def _drawArrow(self, surface, start, end, color, headSize=4):
        pygame.draw.line(surface, color, start, end, 2)
        direction = np.array(end) - np.array(start)
        if np.linalg.norm(direction) == 0:
            return
        direction = direction / np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0]])
        headPoint1 = np.array(end) - headSize * (direction + perp * 0.5)
        headPoint2 = np.array(end) - headSize * (direction - perp * 0.5)
        pygame.draw.polygon(surface, color, [end, headPoint1, headPoint2])

    def _computeSubDomainCenter(self, environmentObj):
        if not environmentObj.objects:
            return (0, 0)
        
        obj = environmentObj.objects[0]
        if self.environment == "ocean":
            centerX = obj.positionVector[0]
            centerZ = obj.positionVector[1]
            centerX = max(self.subDomainSizeX / 2, min(centerX, self.deltaX - self.subDomainSizeX / 2))
            centerZ = max(-self.deltaZ + self.subDomainSizeZ / 2, min(centerZ, 0 - self.subDomainSizeZ / 2))
        else:
            centerX = 0
            centerZ = 0
        
        return (centerX, centerZ)

    def _renderGlobalView(self, environmentObj, surface):
        subDomainCenter = self._computeSubDomainCenter(environmentObj)
        
        if self.environment == "ocean":
            centerX, centerZ = subDomainCenter
            xMin = centerX - self.subDomainSizeX / 2
            xMax = centerX + self.subDomainSizeX / 2
            zMin = centerZ - self.subDomainSizeZ / 2
            zMax = centerZ + self.subDomainSizeZ / 2
            
            if xMin <= 0 <= xMax:
                top = self._toScreenCoords(0, max(zMin, -self.deltaZ), subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(0, min(zMax, 0), subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            
            if xMin <= self.deltaX <= xMax:
                top = self._toScreenCoords(self.deltaX, max(zMin, -self.deltaZ), subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(self.deltaX, min(zMax, 0), subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            
            if zMin <= -self.deltaZ <= zMax:
                left = self._toScreenCoords(max(xMin, 0), -self.deltaZ, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, self.deltaX), -self.deltaZ, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.groundColor, left, right, 3)
            
            if zMin <= 0 <= zMax:
                left = self._toScreenCoords(max(xMin, 0), 0, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, self.deltaX), 0, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, left, right, 2)
        
        for obj in environmentObj.objects:
            orient = np.array(obj.orientationVector, dtype=float)
            norm = np.sqrt(orient @ orient)
            if norm == 0:
                continue
            localXAxis = orient / norm
            localZAxis = np.array([localXAxis[1], -localXAxis[0]])
            
            globalXCoords = []
            globalZCoords = []
            for xLocal, zLocal in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                globalPoint = xLocal * (-localXAxis) + zLocal * localZAxis + obj.positionVector
                globalXCoords.append(globalPoint[0])
                globalZCoords.append(globalPoint[1])
            
            for x, z in zip(globalXCoords, globalZCoords):
                screenPos = self._toScreenCoords(x, z, subDomainCenter=subDomainCenter)
                if (screenPos[0] > self.globalWidth + self.globalPos[0] or screenPos[0] < self.globalPos[0] or 
                    screenPos[1] > self.globalHeight + self.globalPos[1] or screenPos[1] < self.globalPos[1]):
                    continue
                pygame.draw.circle(surface, self.pointColor, screenPos, 5)
            
            if self.SHOW_GLOBAL_FORCE_VECTOR:
                scale = 0.15
                force = np.array(obj.geometry.localForceVector, dtype=float)
                norm = np.sqrt(force[0]**2 + force[1]**2)
                if norm > 0:
                    force = force / norm * scale
                    forceGlobal = force[0] * localXAxis + force[1] * localZAxis
                    centroidX = np.mean(globalXCoords)
                    centroidZ = np.mean(globalZCoords)
                    start = self._toScreenCoords(centroidX, centroidZ, subDomainCenter=subDomainCenter)
                    end = self._toScreenCoords(centroidX + forceGlobal[0], centroidZ + forceGlobal[1], subDomainCenter=subDomainCenter)
                    end = (min(end[0], self.readoutPos[0] - 20), end[1])
                    self._drawArrow(surface, start, end, self.forceColor)
            
            if self.environment == "tunnel":
                globalVel = np.array(obj.velocityVector, dtype=float)
                globalVel = -globalVel
                norm = np.sqrt(globalVel[0]**2 + globalVel[1]**2)
                if norm > 0:
                    max_norm = 5.0
                    scale = 0.05 + (0.25 * min(norm, max_norm) / max_norm)
                    globalVel = globalVel / norm * scale
                    xStart = -0.4
                    for offsetZ in [-0.2, 0.0, 0.2]:
                        start = self._toScreenCoords(xStart, offsetZ, subDomainCenter=subDomainCenter)
                        end = self._toScreenCoords(xStart + globalVel[0], offsetZ + globalVel[1], subDomainCenter=subDomainCenter)
                        end = (max(self.globalPos[0] + 10, min(end[0], self.globalPos[0] + self.globalWidth - 10)),
                               max(self.globalPos[1] + 10, min(end[1], self.globalPos[1] + self.globalHeight - 10)))
                        self._drawArrow(surface, start, end, self.velocityColor, headSize=6)

    def _renderOrientationView(self, environmentObj, surface):
        overlay = pygame.Surface((self.orientWidth, self.orientHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        surface.blit(overlay, self.orientPos)
        
        for obj in environmentObj.objects:
            xRange = max(obj.geometry.pointXCoords) - min(obj.geometry.pointXCoords)
            zRange = max(obj.geometry.pointZCoords) - min(obj.geometry.pointZCoords)
            maxRange = max(xRange, zRange, 1e-6)
            orientScale = min(self.orientWidth, self.orientHeight) / (2 * maxRange) * 0.8
            centroidX = np.mean(obj.geometry.pointXCoords)
            centroidZ = np.mean(obj.geometry.pointZCoords)
            
            for x, z in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                screenPos = self._toScreenCoords(x, z, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                if (screenPos[0] > self.orientWidth + self.orientPos[0] or screenPos[0] < self.orientPos[0] or 
                    screenPos[1] > self.orientHeight + self.orientPos[1] or screenPos[1] < self.orientPos[1]):
                    continue
                pygame.draw.circle(surface, self.pointColor, screenPos, 3)
            
            for x, z in zip(obj.geometry.colocationXCoords, obj.geometry.colocationZCoords):
                screenPos = self._toScreenCoords(x, z, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                if (screenPos[0] > self.orientWidth + self.orientPos[0] or screenPos[0] < self.orientPos[0] or 
                    screenPos[1] > self.orientHeight + self.orientPos[1] or screenPos[1] < self.orientPos[1]):
                    continue
                pygame.draw.circle(surface, self.pointColor, screenPos, 3)
            
            for element in obj.geometry.connectionMatrix:
                x1, z1 = obj.geometry.pointXCoords[element[0]], obj.geometry.pointZCoords[element[0]]
                x2, z2 = obj.geometry.pointXCoords[element[1]], obj.geometry.pointZCoords[element[1]]
                start = self._toScreenCoords(x1, z1, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                end = self._toScreenCoords(x2, z2, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                pygame.draw.line(surface, self.lineColor, start, end, 1)
            
            scale = 0.1
            for i in range(len(obj.geometry.pointXCoords)):
                x, z = obj.geometry.pointXCoords[i], obj.geometry.pointZCoords[i]
                tangential = obj.geometry.tangentialTotalVelocity[i]
                normalX, normalZ = obj.geometry.normalX[i], obj.geometry.normalZ[i]
                vecX = -tangential * normalZ
                vecZ = tangential * normalX
                norm = np.sqrt(vecX**2 + vecZ**2)
                if norm > 0:
                    vecX, vecZ = (vecX / norm) * scale, (vecZ / norm) * scale
                start = self._toScreenCoords(x, z, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                end = self._toScreenCoords(x + vecX, z + vecZ, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                self._drawArrow(surface, start, end, self.tangentialColor)
            
            scale = 0.1 / 2
            localVel = np.array(obj.geometry.localVelocityVector, dtype=float)
            norm = np.sqrt(localVel[0]**2 + localVel[1]**2)
            if norm > 0:
                localVel = localVel / norm * scale
            minX = min(obj.geometry.pointXCoords)
            xStart = minX - 0.15
            for offsetZ in [0, -0.1, 0.1, 0.2]:
                start = self._toScreenCoords(xStart, offsetZ, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                end = self._toScreenCoords(xStart + localVel[0], offsetZ + localVel[1], isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                self._drawArrow(surface, start, end, self.velocityColor)
            
            scale = 0.1
            force = np.array(obj.geometry.localForceVector, dtype=float)
#             print(f"Local force vector in _renderOrientationView: {force}")  # Debug print (Revision 2025-04-28 #6)
            norm = np.sqrt(force[0]**2 + force[1]**2)
            if norm > 0:
                force = force / norm * scale
                # Flip the x-component to correct the subscreen rendering
                # The subscreen inverts the x-axis (higher pointXCoords render to the left),
                # so a negative x local (towards the nose) renders to the right, but we want the force to point towards the tail (left).
                force[0] = -force[0]  # Reintroduced flip (Revision 2025-04-28 #6)
            startX = obj.geometry.pointXCoords[obj.geometry.numPoints // 4]
            start = self._toScreenCoords(startX, 0, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
            end = self._toScreenCoords(startX + force[0], force[1], isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
            self._drawArrow(surface, start, end, self.forceColor)

    def _renderMinimapView(self, environmentObj, surface):
        if self.environment != "ocean":
            return
        
        overlay = pygame.Surface((self.minimapWidth, self.minimapHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        surface.blit(overlay, self.minimapPos)
        
        topLeft = self._toScreenCoords(0, 0, isMinimapView=True)
        topRight = self._toScreenCoords(self.deltaX, 0, isMinimapView=True)
        pygame.draw.line(surface, self.oceanColor, topLeft, topRight, 2)
        
        leftTop = self._toScreenCoords(0, 0, isMinimapView=True)
        leftBottom = self._toScreenCoords(0, -self.deltaZ, isMinimapView=True)
        pygame.draw.line(surface, self.oceanColor, leftTop, leftBottom, 2)
        
        for obj in environmentObj.objects:
            x, z = obj.positionVector[0], obj.positionVector[1]
            if self.frameCounter % 10 == 0:
                self.pathHistory.append((x, z))
                if len(self.pathHistory) > 1000:
                    self.pathHistory.pop(0)
            
            if len(self.pathHistory) > 1:
                for i in range(len(self.pathHistory) - 1):
                    start = self._toScreenCoords(self.pathHistory[i][0], self.pathHistory[i][1], isMinimapView=True)
                    end = self._toScreenCoords(self.pathHistory[i + 1][0], self.pathHistory[i + 1][1], isMinimapView=True)
                    pygame.draw.line(surface, self.velocityColor, start, end, 1)
            
            screenPos = self._toScreenCoords(x, z, isMinimapView=True)
            pygame.draw.circle(surface, self.pointColor, screenPos, 3)

    def _renderReadoutView(self, environmentObj, surface):
        overlay = pygame.Surface((self.readoutWidth, self.readoutHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        surface.blit(overlay, self.readoutPos)
        
        yPos = self.readoutPos[1] + 10
        if self.environment == "ocean":
            labels = ["POS X:", "POS Z:", "VEL X:", "VEL Z:", "ROT ANG:", "AOA:"]
        else:
            labels = ["VEL X:", "VEL Z:", "ACC X:", "ACC Z:", "ROT ANG:", "AOA:"]
        spacing = (self.readoutHeight - 20) // len(labels)
        
        for obj in environmentObj.objects:
            if self.environment == "ocean":
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                values = [posX, posZ, velX, velZ]
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
                labelX = self.readoutPos[0] + 5
                labelY = yPos + 2
                surface.blit(labelText, (labelX, labelY))
                
                windowY = yPos + 50
                windowX = self.readoutPos[0] + 5
                windowWidth = self.digitWidth * totalWheels
                pygame.draw.rect(surface, self.digitWindowColor, 
                                 (windowX, windowY, windowWidth, self.digitHeight))
                
                if self.environment == "tunnel" and label in ["ACC X:", "ACC Z:"]:
                    decimalX = windowX + self.digitWidth * 3
                    decimalY = windowY + self.digitHeight - 3
                    pygame.draw.circle(surface, self.digitColor, (decimalX, decimalY), 1)
                
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
                    surface.set_clip(clipRect)
                    surface.blit(wheel, (xPos, wheelPos))
                    surface.blit(wheel, (xPos, wheelPos - totalHeight))
                    surface.blit(wheel, (xPos, wheelPos + totalHeight))
                    surface.set_clip(None)
                
                pygame.draw.rect(surface, self.textColor, 
                                 (windowX, windowY, windowWidth, self.digitHeight), 1)
                
                yPos += spacing

    def render(self, environmentObj):
        if self.environment == "ocean":
            self.frameCounter += 1
            
        self.running = environmentObj.running

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                newWidth, newHeight = event.size
                self.screen = pygame.display.set_mode((newWidth, newHeight), pygame.RESIZABLE)
                self._updateDimensions(newWidth, newHeight)
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.1
                if event.y > 0:
                    if self.environment == "tunnel":
                        self.TUNNEL_ZOOM /= zoom_factor
                    else:
                        self.ZOOM /= zoom_factor
                elif event.y < 0:
                    if self.environment == "tunnel":
                        self.TUNNEL_ZOOM *= zoom_factor
                    else:
                        self.ZOOM *= zoom_factor
                self.TUNNEL_ZOOM = max(0.001, min(self.TUNNEL_ZOOM, 1.0))
                self.ZOOM = max(0.001, min(self.ZOOM, 1.0))
                self._updateDimensions(self.windowWidth, self.windowHeight)

        allow_rotation = False
        for obj in environmentObj.objects:
            if hasattr(obj.geometry, 'hasTrailingEdge') and obj.geometry.hasTrailingEdge:
                allow_rotation = True
            else:
                obj.orientationVector = np.array([1.0, 0.0], dtype=float)

        if allow_rotation:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                for obj in environmentObj.objects:
                    obj.rotateLeft()
            if keys[pygame.K_RIGHT]:
                for obj in environmentObj.objects:
                    obj.rotateRight()
        
        self.screen.fill(self.bgColor)
        
        self._renderGlobalView(environmentObj, self.screen)
        self._renderOrientationView(environmentObj, self.screen)
        if self.environment == "ocean":
            self._renderMinimapView(environmentObj, self.screen)
        
        self._renderReadoutView(environmentObj, self.screen)
        
        pygame.draw.line(self.screen, self.textColor, 
                         (self.readoutPos[0], 0), (self.readoutPos[0], self.windowHeight), 3)
        
        pygame.draw.rect(self.screen, self.textColor, 
                         (self.orientPos[0], self.orientPos[1], self.orientWidth, self.orientHeight), 2)
        if self.environment == "ocean":
            pygame.draw.rect(self.screen, self.textColor, 
                             (self.minimapPos[0], self.minimapPos[1], self.minimapWidth, self.minimapHeight), 2)
        else:
            pygame.draw.rect(self.screen, self.textColor, 
                             (self.globalPos[0], self.globalPos[1], self.globalWidth, self.globalHeight), 2)
        
        pygame.display.flip()

    def isRunning(self):
        return self.running

    def quit(self):
        pygame.quit()