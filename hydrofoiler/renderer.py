import pygame
import numpy as np
import math

class Renderer:
    ZOOM = 0.01
    TUNNEL_ZOOM = 0.01
    SHOW_GLOBAL_FORCE_VECTOR = True

    def __init__(self, windowWidth=1000, windowHeight=800):
        pygame.init()
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.screen = pygame.display.set_mode((windowWidth, windowHeight), pygame.RESIZABLE)
        pygame.display.set_caption("Hydrofoiler")
        self.bgColor = (255, 255, 255)
        self.pointColor = (0, 0, 255)
        self.lineColor = (0, 0, 255)
        self.tangentialColor = (0, 255, 0)
        self.velocityColor = (255, 0, 0)
        self.forceColor = (255, 105, 180)
        self.thrustColor = (0, 0, 255)
        self.asteroidColor = (100, 100, 100)
        self.laserColor = (255, 0, 0)
        self.oceanColor = (0, 100, 255)
        self.groundColor = (139, 69, 19)
        self.orientBgColor = (200, 200, 200, 128)
        self.environment = None
        self.environmentObj = None
        self.zoomFactor = 1.0
        self.globalOffsetX = 0.0
        self.globalOffsetZ = 0.0
        self.prevSubDomainCenter = [0, 0]
        self.deltaX = 100.0
        self.deltaZ = 100.0
        self._updateDimensions(windowWidth, windowHeight)
        self.running = True
        self.firstRender = True

    def _computeGeometryZoom(self, environmentObj):
        maxSize = 0
        for obj in environmentObj.objects:
            xRange = max(obj.geometry.pointXCoords) - min(obj.geometry.pointXCoords)
            zRange = max(obj.geometry.pointZCoords) - min(obj.geometry.pointZCoords)
            maxSize = max(maxSize, xRange, zRange)
        # In Tunnel mode, account for velocity vectors at x = -0.4, z in [-0.2, 0.2]
        if self.environment == "tunnel":
            maxSize = max(maxSize, 0.5)  # Reduced from 0.8 to make the view smaller; includes x = -0.4 and z = ±0.2
        return maxSize * self.zoomFactor

    def _updateDimensions(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        readoutWidth = windowWidth // 6
        self.globalWidth = windowWidth - readoutWidth
        self.globalHeight = windowHeight
        self.globalPos = (0, 0)
        self.orientHeight = int(windowHeight // 2.5)
        self.orientWidth = self.orientHeight
        self.orientPos = (10, windowHeight - self.orientHeight - 10)
        self.minimapWidth = windowWidth // 6
        self.minimapHeight = windowWidth // 6
        self.minimapPos = (0, 0)  # Remove padding
        self.minimapScaleX = self.minimapWidth / self.deltaX if self.deltaX else 1.0
        self.minimapScaleZ = self.minimapHeight / self.deltaZ if self.deltaZ else 1.0

    def resize(self, size):
        self.windowWidth, self.windowHeight = size
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self._updateDimensions(*size)

    def renderMenu(self, mainMenu):
        mainMenu.render(self.screen, geometrySet=self._checkGeometrySet())
        pygame.display.flip()

    def _checkGeometrySet(self):
        # Access Hydrofoiler's geometry through a reference or external check
        # Since Renderer doesn't have direct access, we'll need to pass it from Hydrofoiler
        return True  # Placeholder; actual implementation in Hydrofoiler

    def renderShop(self, shop):
        shop.render(self.screen)
        pygame.display.flip()

    def handleEvent(self, event, ui_manager=None, ui_active=False):
        # Removed dragging logic
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Zoom in
                self.zoomFactor *= 0.95
#                 print(f"Zoom in: zoomFactor={self.zoomFactor}")
            elif event.button == 5:  # Zoom out
                self.zoomFactor *= 1.05
#                 print(f"Zoom out: zoomFactor={self.zoomFactor}")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                self.resetView()
#                 print("View reset via 'h' key")
            # WASD translation
            elif event.key == pygame.K_d:  # Move frame left (show more right)
                self.globalOffsetX -= 0.5 / self.scaleX
            elif event.key == pygame.K_a:  # Move frame right (show more left)
                self.globalOffsetX += 0.5 / self.scaleX
            elif event.key == pygame.K_w:  # Move frame up (show more down)
                self.globalOffsetZ -= 0.5 / self.scaleZ
            elif event.key == pygame.K_s:  # Move frame down (show more up)
                self.globalOffsetZ += 0.5 / self.scaleZ
        if self.environmentObj:
            self.updateViewParameters()
            
            
    def resetView(self):
        """Reset the view parameters without rendering."""
        self.zoomFactor = 1.0
        self.globalOffsetX = 0.0
        self.globalOffsetZ = 0.0
        if self.environmentObj:
            self.prevSubDomainCenter = self._computeSubDomainCenter(self.environmentObj)
            self.updateViewParameters()
#         print("View reset: zoomFactor=1.0, offsets=0.0")

    def renderEnvironment(self, environmentObj):
        self.environment = environmentObj.__class__.__name__.lower()
        self.environmentObj = environmentObj
        if self.environment in ["ocean", "pilot"]:
            self.deltaX = getattr(environmentObj, 'deltaX', 100.0)
            self.deltaZ = getattr(environmentObj, 'deltaZ', 100.0)
            self.minimapScaleX = self.minimapWidth / self.deltaX if self.deltaX else 1.0
            self.minimapScaleZ = self.minimapHeight / self.deltaZ if self.deltaZ else 1.0
        else:
            self.deltaX = 100.0
            self.deltaZ = 100.0
        newSubDomainCenter = self._computeSubDomainCenter(environmentObj)
        if self.environment in ["ocean", "pilot"]:
            deltaCenterX = newSubDomainCenter[0] - self.prevSubDomainCenter[0]
            deltaCenterZ = newSubDomainCenter[1] - self.prevSubDomainCenter[1]
            self.globalOffsetX -= deltaCenterX
            self.globalOffsetZ -= deltaCenterZ
        # Center view on first render
        if self.firstRender:
            self.zoomFactor = 1.0
            self.globalOffsetX = 0.0
            self.globalOffsetZ = 0.0
            self.prevSubDomainCenter = newSubDomainCenter
            self.firstRender = False
#             print("Initial view centered on first render")
        else:
            self.prevSubDomainCenter = newSubDomainCenter
        self.updateViewParameters()
        self.screen.fill(self.bgColor)
        self._renderGlobalView(environmentObj, self.screen)
        self._renderOrientationView(environmentObj, self.screen)
        if self.environment in ["ocean", "pilot"]:
            self._renderMinimapView(environmentObj, self.screen)

    def updateViewParameters(self):
        """Update subdomain sizes and scales based on current zoom and object."""
        if not self.environmentObj:
            return
        zoom = self._computeGeometryZoom(self.environmentObj)
        # Ensure subdomain sizes are reasonable to prevent division by zero
        self.subDomainSizeX = max(1.2 * zoom, 0.05)
        self.subDomainSizeZ = max(1.2 * zoom, 0.05)
        # Protect against extreme zoom values
        if self.zoomFactor < 1e-6:
            self.zoomFactor = 1e-6
        self.scaleX = self.globalWidth / self.subDomainSizeX
        self.scaleZ = self.globalHeight / self.subDomainSizeZ
#         print(f"Updated view: zoomFactor={self.zoomFactor}, scaleX={self.scaleX}, scaleZ={self.scaleZ}")
        
        

    def _toScreenCoords(self, x, z, isGlobalView=False, isOrientView=False, isMinimapView=False, 
                        orientCenter=None, orientScale=None, subDomainCenter=None):
        if isGlobalView:
            if subDomainCenter is not None:
                x = x - subDomainCenter[0]
                z = z - subDomainCenter[1]
            x = (x + self.globalOffsetX) * self.scaleX + self.globalPos[0] + self.globalWidth / 2
            z = -(z + self.globalOffsetZ) * self.scaleZ + self.globalPos[1] + self.globalHeight / 2
        elif isOrientView:
            x = (x - orientCenter[0]) * orientScale + self.orientPos[0] + self.orientWidth / 2
            z = -(z - orientCenter[1]) * orientScale + self.orientPos[1] + self.orientHeight / 2
        elif isMinimapView:
            # Clamp coordinates to domain
            x = max(0, min(self.deltaX, x))
            z = max(-self.deltaZ, min(0, z))
            # Map x: 0 to deltaX (left to right), z: -deltaZ to 0 (bottom to top)
            screenX = (x / self.deltaX) * self.minimapWidth + self.minimapPos[0]
            screenZ = ((0 - z) / self.deltaZ) * self.minimapHeight + self.minimapPos[1]
            return (screenX, screenZ)
        return (x, z)


    def _drawArrow(self, surface, start, end, color, headSize=10):
        pygame.draw.line(surface, color, start, end, 2)
        # Compute the direction of the arrow
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        # Normalize direction
        dx, dy = dx / length, dy / length
        # Compute perpendicular vectors for arrowhead
        perpX, perpY = -dy, dx
        # Draw arrowhead
        headPoint1 = (end[0] - headSize * dx + headSize * perpX / 2, end[1] - headSize * dy + headSize * perpY / 2)
        headPoint2 = (end[0] - headSize * dx - headSize * perpX / 2, end[1] - headSize * dy - headSize * perpY / 2)
        pygame.draw.polygon(surface, color, [end, headPoint1, headPoint2])

    def _computeSubDomainCenter(self, environmentObj):
        if self.environment in ["ocean", "pilot"]:
            for obj in environmentObj.objects:
                return obj.positionVector  # Center on the object's position
        return [0, 0]  # Default to origin for Tunnel mode

    def _renderGlobalView(self, environmentObj, surface):
        subDomainCenter = self._computeSubDomainCenter(environmentObj)
#         print(f"subDomainCenter: {subDomainCenter}")
        if self.environment in ["ocean", "pilot"]:
            centerX, centerZ = subDomainCenter
            xMin = centerX - self.subDomainSizeX / 2
            xMax = centerX + self.subDomainSizeX / 2
            zMin = centerZ - self.subDomainSizeZ / 2
            zMax = centerZ + self.subDomainSizeZ / 2
            if xMin <= 0 <= xMax:
                top = self._toScreenCoords(0, max(zMin, -self.deltaZ), isGlobalView=True, subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(0, min(zMax, 0), isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            if xMin <= self.deltaX <= xMax:
                top = self._toScreenCoords(self.deltaX, max(zMin, -self.deltaZ), isGlobalView=True, subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(self.deltaX, min(zMax, 0), isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            if zMin <= -self.deltaZ <= zMax:
                left = self._toScreenCoords(max(xMin, 0), -self.deltaZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, self.deltaX), -self.deltaZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.groundColor, left, right, 3)
            if zMin <= 0 <= zMax:
                left = self._toScreenCoords(max(xMin, 0), 0, isGlobalView=True, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, self.deltaX), 0, isGlobalView=True, subDomainCenter=subDomainCenter)
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
#             print(f"Object global coordinates: {list(zip(globalXCoords, globalZCoords))}")
#             print(f"Object positionVector: {obj.positionVector}")
            for x, z in zip(globalXCoords, globalZCoords):
                screenPos = self._toScreenCoords(x, z, isGlobalView=True, subDomainCenter=subDomainCenter)
                if (screenPos[0] > self.globalWidth + self.globalPos[0] or screenPos[0] < self.globalPos[0] or
                    screenPos[1] > self.globalHeight + self.globalPos[1] or screenPos[1] < self.globalPos[1]):
                    continue
                pygame.draw.circle(surface, self.pointColor, screenPos, 5)
            if self.SHOW_GLOBAL_FORCE_VECTOR:
                scale = 0.075
                force = np.array(obj.geometry.localForceVector, dtype=float)
                norm = np.sqrt(force[0]**2 + force[1]**2)
                if norm > 0:
                    force = force / norm * scale
                    forceGlobal = force[0] * localXAxis + force[1] * localZAxis
                    centroidX = np.mean(globalXCoords)
                    centroidZ = np.mean(globalZCoords)
                    start = self._toScreenCoords(centroidX, centroidZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                    end = self._toScreenCoords(centroidX + forceGlobal[0], centroidZ + forceGlobal[1], isGlobalView=True, subDomainCenter=subDomainCenter)
                    self._drawArrow(surface, start, end, self.forceColor)
            if self.environment == "pilot":
                scale = 0.075
                thrust = obj.thrustForce
                norm = np.sqrt(thrust[0]**2 + thrust[1]**2)
                if norm > 0:
                    thrust = thrust / norm * scale
                    start = self._toScreenCoords(centroidX, centroidZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                    end = self._toScreenCoords(centroidX + thrust[0], centroidZ + thrust[1], isGlobalView=True, subDomainCenter=subDomainCenter)
                    self._drawArrow(surface, start, end, self.thrustColor)
            if self.environment == "tunnel":
                globalVel = np.array(obj.velocityVector, dtype=float)
                globalVel = -globalVel  # Reflect the flow direction
                norm = np.sqrt(globalVel[0]**2 + globalVel[1]**2)
                if norm > 0:
                    # Dynamic scaling based on magnitude
                    base_scale = 0.05  # Base length for visibility
                    max_norm = 10.0    # Cap for scaling
                    scale = base_scale * (min(norm, max_norm) / max_norm) * 5  # Adjust length with magnitude
                    globalVel = globalVel / norm * scale  # Scale the vector
                    xStart = -0.4
                    for offsetZ in [-0.2, 0.0, 0.2]:
                        start = self._toScreenCoords(xStart, offsetZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                        end = self._toScreenCoords(xStart + globalVel[0], offsetZ + globalVel[1], isGlobalView=True, subDomainCenter=subDomainCenter)
                        self._drawArrow(surface, start, end, self.velocityColor, headSize=6)
        if self.environment == "pilot":
            for asteroid in environmentObj.asteroids:
                screenPos = self._toScreenCoords(asteroid["pos"][0], asteroid["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                radius = asteroid["radius"] * self.scaleX
                pygame.draw.circle(surface, self.asteroidColor, screenPos, max(5, radius))
            for laser in environmentObj.lasers:
                start = self._toScreenCoords(laser["pos"][0], laser["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                end = self._toScreenCoords(laser["pos"][0] + laser["vel"][0] * 0.1, laser["pos"][1] + laser["vel"][1] * 0.1,
                                          isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.laserColor, start, end, 2)

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
            orientCenter = (centroidX, centroidZ)
            for x, z in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                screenPos = self._toScreenCoords(x, z, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                pygame.draw.circle(overlay, self.pointColor, screenPos, 3)
            for element in obj.geometry.connectionMatrix:
                x1, z1 = obj.geometry.pointXCoords[element[0]], obj.geometry.pointZCoords[element[0]]
                x2, z2 = obj.geometry.pointXCoords[element[1]], obj.geometry.pointZCoords[element[1]]
                start = self._toScreenCoords(x1, z1, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                end = self._toScreenCoords(x2, z2, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                pygame.draw.line(overlay, self.lineColor, start, end, 1)
            maxTangential = max(abs(min(obj.geometry.tangentialTotalVelocity)), abs(max(obj.geometry.tangentialTotalVelocity)), 1e-6)
            targetScreenLength = min(self.orientWidth, self.orientHeight) * 0.25
            tangentialScale = (targetScreenLength / (maxTangential * orientScale)) if maxTangential > 0 else 0
            for i in range(len(obj.geometry.pointXCoords)):
                x, z = obj.geometry.pointXCoords[i], obj.geometry.pointZCoords[i]
                tangential = obj.geometry.tangentialTotalVelocity[i]
                normalX, normalZ = obj.geometry.normalX[i], obj.geometry.normalZ[i]
                vecX = -tangential * normalZ * tangentialScale
                vecZ = tangential * normalX * tangentialScale
                start = self._toScreenCoords(x, z, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                end = self._toScreenCoords(x + vecX, z + vecZ, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                self._drawArrow(surface, start, end, self.tangentialColor)
            # Apparent current vectors (red, in middle third with visible stems)
            localVel = np.array(obj.geometry.localVelocityVector, dtype=float)
            norm = np.sqrt(localVel[0]**2 + localVel[1]**2)
            if norm > 0:
                # Increase target length for visible stems
                targetScreenLength = min(self.orientWidth, self.orientHeight) * 0.2 * (maxRange / 1.0)
                targetScreenLength = min(targetScreenLength, self.orientWidth * 0.3)
                velScale = targetScreenLength / (norm * orientScale)
                localVel = localVel * velScale
                # Position in the middle third of the subscreen
                middleThirdXStart = self.orientPos[0] + self.orientWidth / 3
                middleThirdXEnd = self.orientPos[0] + 2 * self.orientWidth / 3
                middleThirdYStart = self.orientPos[1] + self.orientHeight / 3
                middleThirdYEnd = self.orientPos[1] + 2 * self.orientHeight / 3
                # Center the arrows vertically within the middle third
                zSpacing = zRange * 0.3
                offsets = [-zSpacing, 0, zSpacing] if zRange > 0.1 else [-0.03, 0, 0.03]
                # Position xStart to the left within the middle third
                xStart = min(obj.geometry.pointXCoords) - (maxRange * 0.2)
                for offsetZ in offsets:
                    start = self._toScreenCoords(xStart, offsetZ, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                    end = self._toScreenCoords(xStart + localVel[0], offsetZ + localVel[1], isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                    # Clamp to middle third of the subscreen
                    start = (max(middleThirdXStart, min(middleThirdXEnd, start[0])),
                             max(middleThirdYStart, min(middleThirdYEnd, start[1])))
                    end = (max(middleThirdXStart, min(middleThirdXEnd, end[0])),
                           max(middleThirdYStart, min(middleThirdYEnd, end[1])))
                    self._drawArrow(surface, start, end, self.velocityColor, headSize=5)
            force = np.array(obj.geometry.localForceVector, dtype=float)
            norm = np.sqrt(force[0]**2 + force[1]**2)
            if norm > 0:
                targetScreenLength = min(self.orientWidth, self.orientHeight) * 0.1
                forceScale = targetScreenLength / (norm * orientScale)
                force = force * forceScale
                force[0] = -force[0]
            startX = obj.geometry.pointXCoords[obj.geometry.numPoints // 4]
            start = self._toScreenCoords(startX, 0, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
            end = self._toScreenCoords(startX + force[0], force[1], isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
            self._drawArrow(surface, start, end, self.forceColor, headSize=5)  # Fixed by adding 'end' argument


            
    def _renderMinimapView(self, environmentObj, surface):
        overlay = pygame.Surface((self.minimapWidth, self.minimapHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        pygame.draw.line(overlay, self.oceanColor, (0, 0), (self.minimapWidth, 0), 2)
        pygame.draw.line(overlay, self.oceanColor, (0, self.minimapHeight), (self.minimapWidth, self.minimapHeight), 2)
        pygame.draw.line(overlay, self.oceanColor, (0, 0), (0, self.minimapHeight), 2)
        pygame.draw.line(overlay, self.oceanColor, (self.minimapWidth, 0), (self.minimapWidth, self.minimapHeight), 2)
        for obj in environmentObj.objects:
            for x, z in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                screenPos = self._toScreenCoords(x, z, isMinimapView=True)
                pygame.draw.circle(overlay, self.pointColor, screenPos, 1)
            for element in obj.geometry.connectionMatrix:
                x1, z1 = obj.geometry.pointXCoords[element[0]], obj.geometry.pointZCoords[element[0]]
                x2, z2 = obj.geometry.pointXCoords[element[1]], obj.geometry.pointZCoords[element[1]]
                start = self._toScreenCoords(x1, z1, isMinimapView=True)
                end = self._toScreenCoords(x2, z2, isMinimapView=True)
                pygame.draw.line(overlay, self.lineColor, start, end, 1)
            if hasattr(environmentObj, 'pathHistory'):
                for pos in environmentObj.pathHistory:
                    screenPos = self._toScreenCoords(pos[0], pos[1], isMinimapView=True)
                    pygame.draw.circle(overlay, (255, 0, 0), screenPos, 1)
            markerPos = self._toScreenCoords(obj.positionVector[0], obj.positionVector[1], isMinimapView=True)
            pygame.draw.circle(overlay, (255, 105, 180), markerPos, 3)
        # Add asteroids to minimap
        if self.environment == "pilot":
            for asteroid in environmentObj.asteroids:
                screenPos = self._toScreenCoords(asteroid["pos"][0], asteroid["pos"][1], isMinimapView=True)
                radius = asteroid["radius"] * (self.minimapWidth / self.deltaX)  # Scale radius to minimap
                pygame.draw.circle(overlay, self.asteroidColor, screenPos, max(2, radius))
        surface.blit(overlay, self.minimapPos)

    def quit(self):
        pygame.quit()