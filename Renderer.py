import pygame
import numpy as np
import math

class Renderer:
    ZOOM = 0.01  # Zoom factor for Ocean global view
    TUNNEL_ZOOM = 0.01  # Zoom factor for Tunnel global view
    SHOW_GLOBAL_FORCE_VECTOR = True  # Boolean to control display of global frame force vector

    def __init__(self, windowWidth=800, windowHeight=600, deltaX=None, deltaZ=None, environment="ocean"):
        """
        Initialize the Pygame renderer for Ocean or Tunnel with subviews.
        
        Args:
            windowWidth (int): Width of the Pygame window.
            windowHeight (int): Height of the Pygame window.
            deltaX (float): X-axis range for the global frame (Ocean only, covers [0, deltaX]).
            deltaZ (float): Z-axis range for the global frame (Ocean only, covers [0, -deltaZ]).
            environment (str): "ocean" or "tunnel" to determine rendering mode.
        """
        pygame.init()
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.environment = environment  # "ocean" or "tunnel"
        
        # Ocean-specific attributes
        if self.environment == "ocean":
            if deltaX is None or deltaZ is None:
                raise ValueError("deltaX and deltaZ must be provided for Ocean environment")
            self.deltaX = deltaX  # Covers x: [0, deltaX]
            self.deltaZ = deltaZ  # Covers z: [0, -deltaZ]
        else:
            self.deltaX = None
            self.deltaZ = None
        
        # Make the window resizable
        self.screen = pygame.display.set_mode((windowWidth, windowHeight), pygame.RESIZABLE)
        pygame.display.set_caption("Object Visualization")
        
        # Define digit height, font, and colors before creating digit wheels
        self.digitHeight = 20
        self.font = pygame.font.SysFont("arial", 14)
        
        # Colors
        self.bgColor = (255, 255, 255)  # White
        self.pointColor = (0, 0, 255)  # Blue
        self.lineColor = (0, 0, 255)  # Blue for connection lines
        self.tangentialColor = (0, 255, 0)  # Green
        self.velocityColor = (255, 0, 0)  # Red
        self.forceColor = (255, 105, 180)  # Pink
        self.textColor = (0, 0, 0)  # Black
        self.oceanColor = (0, 100, 255)  # Blue for ocean boundaries
        self.groundColor = (139, 69, 19)  # Brown for ground
        self.orientBgColor = (200, 200, 200, 128)  # Semi-transparent gray for orientation subview
        self.digitBgColor = (50, 50, 50)  # Dark gray for digit windows
        self.digitColor = (255, 255, 255)  # White for digit text
        self.digitWindowColor = (30, 30, 30)  # Darker gray for digit window background
        
        # Initialize subview dimensions and scaling factors
        self._updateDimensions(windowWidth, windowHeight)
        
        self.running = True
        
        # Digit wheel setup for readouts
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]  # 3 digit wheels (for hundreds, tens, ones)
        self.signWheel = self._createSignWheel()  # 1 sign wheel
        self.wheelOffsets = {}  # Store offsets for smooth animation
        self.animationSpeed = 15  # Speed for transitions (pixels per frame)
        self.animationDuration = 10  # Number of frames for animation (approx 166ms at 60 FPS)
        self.animationFrames = {}  # Track animation progress for each wheel
        self.prevValues = {}  # Store previous values for smooth transitions
        
        # Path history for minimap (Ocean only)
        if self.environment == "ocean":
            self.pathHistory = []  # List of (x, z) positions
            self.frameCounter = 0  # For path history updates
        
        # Smoothing for AoA
        self.prevAoa = {}  # Store previous AoA values for smoothing
        self.smoothingFactor = 0.1  # Smoothing factor for AoA

    def _updateDimensions(self, windowWidth, windowHeight):
        """
        Update subview dimensions and scaling factors when the window is resized.
        
        Args:
            windowWidth (int): New width of the window.
            windowHeight (int): New height of the window.
        """
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        
        # Readout dimensions (same for both Ocean and Tunnel)
        self.readoutWidth = windowWidth // 6
        self.readoutHeight = windowHeight - 20
        self.readoutPos = (windowWidth - self.readoutWidth - 10, 10)  # Right side
        
        # Global and Orientation view dimensions and positions
        if self.environment == "ocean":
            # Ocean: Global frame is main view, Orientation is subscreen
            self.globalWidth = windowWidth - self.readoutWidth - 20  # Exclude readout area
            self.globalHeight = windowHeight
            self.globalPos = (0, 0)
            
            self.orientWidth = windowWidth // 4
            self.orientHeight = windowWidth // 4
            self.orientPos = (10, windowHeight - self.orientHeight - 10)  # Bottom-left
        else:  # Tunnel
            # Tunnel: Orientation view is main view, Global frame is subscreen
            self.orientWidth = windowWidth - self.readoutWidth - 20  # Exclude readout area
            self.orientHeight = windowHeight
            self.orientPos = (0, 0)  # Main view position
            
            self.globalWidth = windowWidth // 4
            self.globalHeight = windowWidth // 4
            self.globalPos = (10, windowHeight - self.globalHeight - 10)  # Bottom-left
        
        # Minimap dimensions (Ocean only)
        if self.environment == "ocean":
            self.minimapWidth = windowWidth // 4
            self.minimapHeight = windowWidth // 4
            self.minimapPos = (10, 10)  # Top-left
            # Minimap scaling (covers entire domain)
            self.minimapScaleX = self.minimapWidth / self.deltaX
            self.minimapScaleZ = self.minimapHeight / self.deltaZ
        
        # Magnified subset for global view
        zoom = self.TUNNEL_ZOOM if self.environment == "tunnel" else self.ZOOM
        # Use a fixed domain size for Tunnel since no deltaX/deltaZ is provided
        domain_size = 100.0  # Arbitrary default domain size for Tunnel
        self.subDomainSizeX = domain_size * zoom  # e.g., 1 if domain_size=100 and ZOOM=0.01
        self.subDomainSizeZ = domain_size * zoom  # e.g., 1 if domain_size=100 and ZOOM=0.01
        self.scaleX = self.globalWidth / self.subDomainSizeX
        self.scaleZ = self.globalHeight / self.subDomainSizeZ
        
        # Update digit width for readouts
        self.digitWidth = (self.readoutWidth - 60) // 4  # 4 wheels: sign + 3 digits
        # Recreate digit wheels with new width
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]
        self.signWheel = self._createSignWheel()

    def _createDigitWheel(self):
        """
        Create a digit wheel surface for the altimeter-style readout.
        
        Returns:
            pygame.Surface: Surface with digits 0-9 arranged vertically.
        """
        wheelHeight = self.digitHeight * 10  # 10 digits
        wheel = pygame.Surface((self.digitWidth, wheelHeight), pygame.SRCALPHA)
        for i in range(10):
            digitText = self.font.render(str(i), True, self.digitColor)
            xPos = (self.digitWidth - digitText.get_width()) // 2
            yPos = i * self.digitHeight + (self.digitHeight - digitText.get_height()) // 2
            wheel.blit(digitText, (xPos, yPos))
        return wheel

    def _createSignWheel(self):
        """
        Create a sign wheel surface for the altimeter-style readout (+/-).
        
        Returns:
            pygame.Surface: Surface with + and - symbols arranged vertically.
        """
        wheelHeight = self.digitHeight * 2  # 2 symbols (+ and -)
        wheel = pygame.Surface((self.digitWidth, wheelHeight), pygame.SRCALPHA)
        for i, symbol in enumerate(["+", "-"]):
            symbolText = self.font.render(symbol, True, self.digitColor)
            xPos = (self.digitWidth - symbolText.get_width()) // 2
            yPos = i * self.digitHeight + (self.digitHeight - symbolText.get_height()) // 2
            wheel.blit(symbolText, (xPos, yPos))
        return wheel

    def _toScreenCoords(self, x, z, isOrientView=False, isMinimapView=False, orientCenter=(0, 0), orientScale=1.0, subDomainCenter=(0, 0)):
        """
        Convert coordinates to Pygame screen coordinates.
        
        Args:
            x (float): X-coordinate in plot space (global or local).
            z (float): Z-coordinate in plot space (global or local).
            isOrientView (bool): If True, map to orientation subview local frame.
            isMinimapView (bool): If True, map to minimap view (entire domain, Ocean only).
            orientCenter (tuple): Center point for local frame in plot coordinates.
            orientScale (float): Scaling factor for local frame.
            subDomainCenter (tuple): (centerX, centerZ) for global view magnified subset.
        
        Returns:
            tuple: (screenX, screenY) in Pygame coordinates.
        """
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
            zNormalized = (z - zRangeMin) / (zRangeMax - zRangeMin)  # [0, 1]
            screenZ = self.globalPos[1] + (1 - zNormalized) * self.globalHeight  # Invert: [0, 1] -> [globalHeight, 0]
        return (screenX, screenZ)

    def _drawArrow(self, surface, start, end, color, headSize=4):
        """
        Draw an arrow from start to end with a triangular head.
        
        Args:
            surface (pygame.Surface): Surface to draw on.
            start (tuple): Starting point (x, y) in screen coordinates.
            end (tuple): Ending point (x, y) in screen coordinates.
            color (tuple): RGB color.
            headSize (float): Size of the arrowhead.
        """
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
        """
        Compute the center of the magnified subset based on the object's position (Ocean) or fixed center (Tunnel).
        
        Args:
            environmentObj: Ocean or Tunnel instance with self.objects.
        
        Returns:
            tuple: (centerX, centerZ) for the magnified subset.
        """
        if not environmentObj.objects:
            return (0, 0)  # Default center if no objects
        
        obj = environmentObj.objects[0]
        if self.environment == "ocean":
            centerX = obj.positionVector[0]
            centerZ = obj.positionVector[1]
            centerX = max(self.subDomainSizeX / 2, min(centerX, self.deltaX - self.subDomainSizeX / 2))
            centerZ = max(-self.deltaZ + self.subDomainSizeZ / 2, min(centerZ, 0 - self.subDomainSizeZ / 2))
        else:
            # For Tunnel, object is stationary, so center is fixed at (0, 0)
            centerX = 0
            centerZ = 0
        
        return (centerX, centerZ)

    def _renderGlobalView(self, environmentObj, surface):
        """
        Render the global view (main view for Ocean, subscreen for Tunnel).
        
        Args:
            environmentObj: Ocean or Tunnel instance with self.objects.
            surface (pygame.Surface): Surface to render on.
        """
        subDomainCenter = self._computeSubDomainCenter(environmentObj)
        
        # Draw semi-transparent background for Tunnel global view (now subscreen)
        if self.environment == "tunnel":
            overlay = pygame.Surface((self.globalWidth, self.globalHeight), pygame.SRCALPHA)
            overlay.fill(self.orientBgColor)
            surface.blit(overlay, self.globalPos)
        
        # Draw domain boundaries for Ocean only
        if self.environment == "ocean":
            centerX, centerZ = subDomainCenter
            xMin = centerX - self.subDomainSizeX / 2
            xMax = centerX + self.subDomainSizeX / 2
            zMin = centerZ - self.subDomainSizeZ / 2
            zMax = centerZ + self.subDomainSizeZ / 2
            
            if xMin <= 0 <= xMax:  # Left boundary (x=0)
                top = self._toScreenCoords(0, max(zMin, -self.deltaZ), subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(0, min(zMax, 0), subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            
            if xMin <= self.deltaX <= xMax:  # Right boundary (x=deltaX)
                top = self._toScreenCoords(self.deltaX, max(zMin, -self.deltaZ), subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(self.deltaX, min(zMax, 0), subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            
            if zMin <= -self.deltaZ <= zMax:  # Bottom boundary (z=-deltaZ)
                left = self._toScreenCoords(max(xMin, 0), -self.deltaZ, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, self.deltaX), -self.deltaZ, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.groundColor, left, right, 3)
            
            if zMin <= 0 <= zMax:  # Top boundary (z=0)
                left = self._toScreenCoords(max(xMin, 0), 0, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, self.deltaX), 0, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, left, right, 2)
        
        for obj in environmentObj.objects:
            orient = np.array(obj.orientationVector, dtype=float)
            norm = np.sqrt(orient @ orient)
            if norm == 0:
                continue
            localXAxis = orient / norm
            localZAxis = np.array([localXAxis[1], -localXAxis[0]])  # Counterclockwise rotation to match projectForceVector
            
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
            
            # Draw global force vector if enabled
            if self.SHOW_GLOBAL_FORCE_VECTOR:
                scale = 0.1  # Same scale as velocity vector in orientation view for visibility
                force = np.array(obj.geometry.localForceVector, dtype=float)
                norm = np.sqrt(force[0]**2 + force[1]**2)
                if norm > 0:
                    force = force / norm * scale
                    # Transform local force vector to global coordinates
                    forceGlobal = force[0] * (-localXAxis) + force[1] * localZAxis
                    # Use object's centroid position as the starting point
                    centroidX = np.mean(globalXCoords)
                    centroidZ = np.mean(globalZCoords)
                    start = self._toScreenCoords(centroidX, centroidZ, subDomainCenter=subDomainCenter)
                    end = self._toScreenCoords(centroidX + forceGlobal[0], centroidZ + forceGlobal[1], subDomainCenter=subDomainCenter)
                    self._drawArrow(surface, start, end, self.forceColor)
            
            # Draw global velocity arrows for Tunnel (now in subscreen)
            if self.environment == "tunnel":
                scale = 0.1  # Same scale as other vectors
                globalVel = np.array(obj.velocityVector, dtype=float)
                globalVel = -globalVel  # Invert to show apparent current
                norm = np.sqrt(globalVel[0]**2 + globalVel[1]**2)
                if norm > 0:
                    globalVel = globalVel / norm * scale
                    # Draw three arrows vertically spaced on the left side
                    xStart = -0.5  # Position on the left side of the global view
                    for offsetZ in [-0.2, 0.0, 0.2]:  # Three arrows at different heights
                        start = self._toScreenCoords(xStart, offsetZ, subDomainCenter=subDomainCenter)
                        end = self._toScreenCoords(xStart + globalVel[0], offsetZ + globalVel[1], subDomainCenter=subDomainCenter)
                        self._drawArrow(surface, start, end, self.velocityColor, headSize=6)

    def _renderOrientationView(self, environmentObj, surface):
        """
        Render the orientation view (subscreen for Ocean, main view for Tunnel).
        
        Args:
            environmentObj: Ocean or Tunnel instance with self.objects.
            surface (pygame.Surface): Surface to render on.
        """
        # Draw semi-transparent background for Ocean orientation view (subscreen)
        if self.environment == "ocean":
            overlay = pygame.Surface((self.orientWidth, self.orientHeight), pygame.SRCALPHA)
            overlay.fill(self.orientBgColor)
            surface.blit(overlay, self.orientPos)
        
        for obj in environmentObj.objects:
            xRange = max(obj.geometry.pointXCoords) - min(obj.geometry.pointXCoords)
            zRange = max(obj.geometry.pointZCoords) - min(obj.geometry.pointZCoords)
            maxRange = max(xRange, zRange, 1e-6)  # Avoid division by zero
            orientScale = min(self.orientWidth, self.orientHeight) / (2 * maxRange) * 0.8  # Fit within 80% of subview
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
            minX = min(obj.geometry.pointXCoords) - 0.1
            for offsetZ in [0, -0.1, 0.1, 0.2]:
                start = self._toScreenCoords(minX, offsetZ, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                end = self._toScreenCoords(minX + localVel[0], offsetZ + localVel[1], isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
                self._drawArrow(surface, start, end, self.velocityColor)
            
            scale = 0.1  # 10x larger
            force = np.array(obj.geometry.localForceVector, dtype=float)
            norm = np.sqrt(force[0]**2 + force[1]**2)
            if norm > 0:
                force = force / norm * scale
            startX = obj.geometry.pointXCoords[obj.geometry.numPoints // 4]
            start = self._toScreenCoords(startX, 0, isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
            end = self._toScreenCoords(startX + force[0], force[1], isOrientView=True, orientCenter=(centroidX, centroidZ), orientScale=orientScale)
            self._drawArrow(surface, start, end, self.forceColor)

    def _renderMinimapView(self, environmentObj, surface):
        """
        Render the minimap view in the top-left corner (Ocean only).
        
        Args:
            environmentObj: Ocean instance with self.objects.
            surface (pygame.Surface): Surface to render on (minimap subview).
        """
        if self.environment != "ocean":
            return  # Skip minimap for Tunnel
        
        # Draw semi-transparent background
        overlay = pygame.Surface((self.minimapWidth, self.minimapHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        surface.blit(overlay, self.minimapPos)
        
        # Draw ocean boundaries (only top and left)
        # Top (z=0)
        topLeft = self._toScreenCoords(0, 0, isMinimapView=True)
        topRight = self._toScreenCoords(self.deltaX, 0, isMinimapView=True)
        pygame.draw.line(surface, self.oceanColor, topLeft, topRight, 2)
        
        # Left (x=0)
        leftTop = self._toScreenCoords(0, 0, isMinimapView=True)
        leftBottom = self._toScreenCoords(0, -self.deltaZ, isMinimapView=True)
        pygame.draw.line(surface, self.oceanColor, leftTop, leftBottom, 2)
        
        # Update path history and render object position
        for obj in environmentObj.objects:
            x, z = obj.positionVector[0], obj.positionVector[1]
            # Add current position to path history (every 10 frames to reduce density)
            if self.frameCounter % 10 == 0:
                self.pathHistory.append((x, z))
                # Limit path history to prevent excessive memory usage
                if len(self.pathHistory) > 1000:
                    self.pathHistory.pop(0)
            
            # Draw path history as a line
            if len(self.pathHistory) > 1:
                for i in range(len(self.pathHistory) - 1):
                    start = self._toScreenCoords(self.pathHistory[i][0], self.pathHistory[i][1], isMinimapView=True)
                    end = self._toScreenCoords(self.pathHistory[i + 1][0], self.pathHistory[i + 1][1], isMinimapView=True)
                    pygame.draw.line(surface, self.velocityColor, start, end, 1)
            
            # Draw marker for current position (no flashing)
            screenPos = self._toScreenCoords(x, z, isMinimapView=True)
            pygame.draw.circle(surface, self.pointColor, screenPos, 3)

    def _renderReadoutView(self, environmentObj, surface):
        """
        Render the readout view on the right side with altimeter-style digit wheels.
        
        Args:
            environmentObj: Ocean or Tunnel instance with self.objects.
            surface (pygame.Surface): Surface to render on (readout subview).
        """
        overlay = pygame.Surface((self.readoutWidth, self.readoutHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        surface.blit(overlay, self.readoutPos)
        
        yPos = self.readoutPos[1] + 20
        if self.environment == "ocean":
            labels = ["Pos X", "Pos Z", "Vel X", "Vel Z", "Rot Ang", "AoA"]
        else:  # Tunnel
            labels = ["Vel X", "Vel Z", "Acc X", "Acc Z", "Rot Ang", "AoA"]
        spacing = (self.readoutHeight - 40) // len(labels)  # Even spacing
        
        for obj in environmentObj.objects:
            if self.environment == "ocean":
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                values = [posX, posZ, velX, velZ]
            else:  # Tunnel
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                accX, accZ = obj.accelerationVector[0], obj.accelerationVector[1]
                values = [velX, velZ, accX, accZ]
            
            # Rotation angle (relative to [0, -1])
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
            
            # Angle of attack (between orientation and velocity)
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
                # Apply smoothing to AoA
                key = f"{id(obj)}_AoA"
                if key in self.prevAoa:
                    aoa = self.prevAoa[key] + self.smoothingFactor * (aoa - self.prevAoa[key])
                self.prevAoa[key] = aoa
            else:
                aoa = 0.0
            
            values.extend([rotAngle, aoa])  # Append Rot Ang and AoA to the values list
            
            for label, value in zip(labels, values):
                # Smooth the value to ensure animation
                key = f"{id(obj)}_{label}_value"
                if key not in self.prevValues:
                    self.prevValues[key] = value
                smoothedValue = self.prevValues[key] + self.smoothingFactor * (value - self.prevValues[key])
                self.prevValues[key] = smoothedValue
                value = smoothedValue
                
                # All readouts: sign + 3 digits (e.g., "+062" or "+180")
                numDigits = 3
                if label in ["Rot Ang", "AoA"]:
                    maxValue = 180
                    minValue = -180
                    value = max(min(value, maxValue), minValue)
                    valueStr = f"{abs(int(value)):03d}"  # e.g., 123 -> "123"
                else:
                    maxValue = 999
                    minValue = -999
                    value = max(min(value, maxValue), minValue)
                    # Extract digits for whole number
                    absValue = abs(value)
                    hundreds = int(absValue // 100) % 10
                    tens = int(absValue // 10) % 10
                    ones = int(absValue) % 10
                    valueStr = f"{hundreds:01d}{tens:01d}{ones:01d}"
                
                signIdx = 0 if value >= 0 else 1  # 0 for "+", 1 for "-"
                totalWheels = 4  # Sign + 3 digits
                
                # Initialize wheel offsets
                key = f"{id(obj)}_{label}"
                if key not in self.wheelOffsets:
                    self.wheelOffsets[key] = [0] * totalWheels  # Sign + 3 digits
                if key not in self.animationFrames:
                    self.animationFrames[key] = [0] * totalWheels  # Animation frame counters
                
                # Draw label with fixed spacing
                labelText = self.font.render(f"{label}:", True, self.textColor)
                labelX = self.readoutPos[0] + 10
                labelY = yPos + self.digitHeight // 2 - labelText.get_height() // 2
                surface.blit(labelText, (labelX, labelY))
                
                # Draw digit window background
                windowX = self.readoutPos[0] + 50
                windowWidth = self.digitWidth * totalWheels
                pygame.draw.rect(surface, self.digitWindowColor, 
                                 (windowX, yPos, windowWidth, self.digitHeight))
                
                # Prepare wheels to draw: sign + 3 digits
                wheelsToDraw = [(self.signWheel, signIdx, 2)]  # Sign wheel with 2 positions
                for i, digit in enumerate(valueStr):
                    wheelsToDraw.append((self.digitWheels[i], int(digit), 10))
                
                # Draw all wheels
                for i, (wheel, targetDigit, numPositions) in enumerate(wheelsToDraw):
                    xPos = windowX + i * self.digitWidth
                    currentOffset = self.wheelOffsets[key][i]
                    targetOffset = targetDigit * self.digitHeight
                    
                    # Optimize spinning direction
                    diff = targetOffset - currentOffset
                    totalHeight = self.digitHeight * numPositions
                    if diff > totalHeight / 2:
                        diff -= totalHeight
                    elif diff < -totalHeight / 2:
                        diff += totalHeight
                    
                    # Animate the wheel
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
                    
                    # Compute wheel position for smooth animation
                    wheelPos = yPos - currentOffset
                    
                    # Ensure wrapping for continuous animation
                    clipRect = pygame.Rect(xPos, yPos, self.digitWidth, self.digitHeight)
                    surface.set_clip(clipRect)
                    surface.blit(wheel, (xPos, wheelPos))
                    surface.blit(wheel, (xPos, wheelPos - totalHeight))
                    surface.blit(wheel, (xPos, wheelPos + totalHeight))
                    surface.set_clip(None)
                
                # Draw digit window border
                pygame.draw.rect(surface, self.textColor, 
                                 (windowX, yPos, windowWidth, self.digitHeight), 1)
                
                yPos += spacing

    def render(self, environmentObj):
        """
        Render the environment's objects in one Pygame window with subviews.
        
        Args:
            environmentObj: Ocean or Tunnel instance with self.objects.
        """
        # Increment frame counter for path history (Ocean only)
        if self.environment == "ocean":
            self.frameCounter += 1
            
        self.running = environmentObj.running

        # Handle Pygame events for window closure and resizing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                newWidth, newHeight = event.size
                self.screen = pygame.display.set_mode((newWidth, newHeight), pygame.RESIZABLE)
                self._updateDimensions(newWidth, newHeight)
        
        # Handle continuous key presses for rotation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            for obj in environmentObj.objects:
                obj.rotateLeft()  # Left key rotates left (counterclockwise)
        if keys[pygame.K_RIGHT]:
            for obj in environmentObj.objects:
                obj.rotateRight()  # Right key rotates right (clockwise)
        
        # Clear screen
        self.screen.fill(self.bgColor)
        
        # Render views based on environment
        if self.environment == "ocean":
            # Ocean: Global view is main, Orientation is subscreen
            self._renderGlobalView(environmentObj, self.screen)
            self._renderOrientationView(environmentObj, self.screen)
            self._renderMinimapView(environmentObj, self.screen)
        else:
            # Tunnel: Orientation view is main, Global view is subscreen
            self._renderOrientationView(environmentObj, self.screen)
            self._renderGlobalView(environmentObj, self.screen)
        
        # Render readout view (right side)
        self._renderReadoutView(environmentObj, self.screen)
        
        # Draw borders for subviews
        pygame.draw.rect(self.screen, self.textColor, 
                         (self.orientPos[0], self.orientPos[1], self.orientWidth, self.orientHeight), 2)
        pygame.draw.rect(self.screen, self.textColor, 
                         (self.readoutPos[0], self.readoutPos[1], self.readoutWidth, self.readoutHeight), 2)
        if self.environment == "ocean":
            pygame.draw.rect(self.screen, self.textColor, 
                             (self.minimapPos[0], self.minimapPos[1], self.minimapWidth, self.minimapHeight), 2)
        else:
            pygame.draw.rect(self.screen, self.textColor, 
                             (self.globalPos[0], self.globalPos[1], self.globalWidth, self.globalHeight), 2)
        
        # Update display
        pygame.display.flip()

    def isRunning(self):
        """
        Check if the Pygame window is still running.
        
        Returns:
            bool: True if the window is open, False otherwise.
        """
        return self.running

    def quit(self):
        """Close the Pygame window and clean up."""
        pygame.quit()