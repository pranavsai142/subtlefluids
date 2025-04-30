import pygame
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import copy
from geometry import Geometry, Circle, Foil

MAX_FRAMES = 1000
PLOT_FRAMES = False
GRAVITY = 9.81
GRAVITY_VECTOR = np.array([0, -GRAVITY])
DELTA_T = 0.01
MAX_VELOCITY = 1000000
MAX_ACCELERATION = 1000000
DELTA_ROTATION = 2.0
THRUST_STRENGTH = 1000

class Pilot:
    def __init__(self, deltaX, deltaZ):
        self.deltaX = deltaX
        self.deltaZ = deltaZ
        self.objects = []
        self.asteroids = []
        self.lasers = []
        self.frameNumber = 0
        self.frameFilenames = []
        self.orientationFrameFilenames = []
        self.objectFrameFilenames = []
        self.forceVectors = []
        self.thrustVectors = []
        self.positionVectors = []
        self.velocityVectors = []
        self.accelerationVectors = []
        self.addedMasses = []
        self.pathHistory = []
        self.running = True
        self.maxHistoryLength = 100
        self.spawnAsteroids()

    def addObject(self, geometryData):
        object = Object(geometryData, self.deltaX/2, -self.deltaZ/2)
        object.pointRight()
        self.objects.append(object)
        return object

    def spawnAsteroids(self):
        for _ in range(5):
            x = np.random.uniform(0, self.deltaX)
            z = np.random.uniform(-self.deltaZ, 0)
            vx = np.random.uniform(-1, 1)
            vz = np.random.uniform(-1, 1)
            radius = np.random.uniform(0.5, 2)
            self.asteroids.append({"pos": [x, z], "vel": [vx, vz], "radius": radius})


    def shootLaser(self):
        if self.objects:
            obj = self.objects[0]
            pos = np.array(obj.positionVector)
            orient = np.array(obj.orientationVector)
            norm = np.sqrt(orient @ orient)
            if norm > 0:
                direction = orient / norm
                self.lasers.append({"pos": pos.copy(), "vel": direction * 10, "lifetime": 100})

    def handleKey(self, event):
        if event.key == pygame.K_UP:
            for obj in self.objects:
                obj.thrustForward()
        elif event.key == pygame.K_DOWN:
            for obj in self.objects:
                obj.thrustBackward()
        else:
            for obj in self.objects:
                obj.disableThrust()
        if event.key == pygame.K_SPACE:
            self.shootLaser()
        elif event.key == pygame.K_LEFT:
            for obj in self.objects:
                obj.rotateLeft()
        elif event.key == pygame.K_RIGHT:
            for obj in self.objects:
                obj.rotateRight()
                
    def handleKeys(self, keys):
        if keys[pygame.K_SPACE]:
            self.shootLaser()
        if keys[pygame.K_LEFT]:
            for obj in self.objects:
                obj.rotateLeft()
        if keys[pygame.K_RIGHT]:
            for obj in self.objects:
                obj.rotateRight()
        if keys[pygame.K_UP]:
            for obj in self.objects:
                obj.thrustForward()
        elif keys[pygame.K_DOWN]:
            for obj in self.objects:
                obj.thrustBackward()
        else:
            for obj in self.objects:
                obj.disableThrust()

    def advanceTime(self):
#         self.thrustForce = np.array([0.0, 0.0], dtype=np.float64)
        for obj in self.objects:
            obj.updatePosition()
            self.pathHistory.append(obj.positionVector.copy())
            if len(self.pathHistory) > self.maxHistoryLength:
                self.pathHistory.pop(0)
            if self.isOutsideBounds(obj) or self.frameNumber >= MAX_FRAMES:
                self.running = False
                if PLOT_FRAMES:
                    self.createMovie("pilot_global_forces.gif", self.frameFilenames)
                    self.createMovie("pilot_local_forces.gif", self.objectFrameFilenames)
                    self.createMovie("pilot_local_orientations.gif", self.orientationFrameFilenames)
                    self.plotVectorTimeseries("pilot_forces_timeseries.png", self.forceVectors, "force")
                    self.plotVectorTimeseries("pilot_thrust_timeseries.png", self.thrustVectors, "thrust")
                    self.plotVectorTimeseries("pilot_position_timeseries.png", self.positionVectors, "position")
                    self.plotVectorTimeseries("pilot_velocity_timeseries.png", self.velocityVectors, "velocity")
                    self.plotVectorTimeseries("pilot_acceleration_timeseries.png", self.accelerationVectors, "acceleration")
                    self.plotTimeseries("pilot_added_mass_timeseries.png", self.addedMasses, "added mass")
            else:
                if PLOT_FRAMES:
                    self.forceVectors.append(obj.forceVector)
                    self.thrustVectors.append(obj.thrustForce)
                    self.positionVectors.append(copy.copy(obj.positionVector))
                    self.velocityVectors.append(copy.copy(obj.velocityVector))
                    self.accelerationVectors.append(copy.copy(obj.accelerationVector))
                    self.addedMasses.append(copy.copy(obj.addedMass))
                    if len(self.forceVectors) > self.maxHistoryLength:
                        self.forceVectors.pop(0)
                    if len(self.thrustVectors) > self.maxHistoryLength:
                        self.thrustVectors.pop(0)
                    if len(self.positionVectors) > self.maxHistoryLength:
                        self.positionVectors.pop(0)
                    if len(self.velocityVectors) > self.maxHistoryLength:
                        self.velocityVectors.pop(0)
                    if len(self.accelerationVectors) > self.maxHistoryLength:
                        self.accelerationVectors.pop(0)
                    if len(self.addedMasses) > self.maxHistoryLength:
                        self.addedMasses.pop(0)
                    frameFilename = os.path.join("graphs", f"pilot_global_forces_{self.frameNumber}.png")
                    orientationFrameFilename = os.path.join("graphs", f"pilot_local_orientation_{self.frameNumber}.png")
                    self.plotForces(frameFilename, orientationFrameFilename, obj)
                    self.frameFilenames.append(frameFilename)
                    self.orientationFrameFilenames.append(orientationFrameFilename)
                    objectFrameFilename = os.path.join("graphs", f"pilot_local_forces_{self.frameNumber}.png")
                    obj.geometry.plotForces(objectFrameFilename, obj.geometry.localVelocityVector,
                                           obj.geometry.tangentialTotalVelocity, obj.geometry.localForceVector)
                    self.objectFrameFilenames.append(objectFrameFilename)
                    if len(self.frameFilenames) > self.maxHistoryLength:
                        self.frameFilenames.pop(0)
                    if len(self.orientationFrameFilenames) > self.maxHistoryLength:
                        self.orientationFrameFilenames.pop(0)
                    if len(self.objectFrameFilenames) > self.maxHistoryLength:
                        self.objectFrameFilenames.pop(0)
                self.frameNumber += 1
        self.updateAsteroids()
        self.updateLasers()
        
        
    def updateAsteroids(self):
        for asteroid in self.asteroids:
            asteroid["pos"][0] += asteroid["vel"][0] * DELTA_T
            asteroid["pos"][1] += asteroid["vel"][1] * DELTA_T
            if asteroid["pos"][0] < 0 or asteroid["pos"][0] > self.deltaX:
                asteroid["vel"][0] = -asteroid["vel"][0]
            if asteroid["pos"][1] < -self.deltaZ or asteroid["pos"][1] > 0:
                asteroid["vel"][1] = -asteroid["vel"][1]
                
    def cleanup(self):
        self.pathHistory.clear()
        self.asteroids.clear()
        self.lasers.clear()
        self.frameFilenames.clear()
        self.orientationFrameFilenames.clear()
        self.objectFrameFilenames.clear()
        self.forceVectors.clear()
        self.thrustVectors.clear()
        self.positionVectors.clear()
        self.velocityVectors.clear()
        self.accelerationVectors.clear()
        self.addedMasses.clear()
        plt.close('all')

    def updateLasers(self):
        newLasers = []
        for laser in self.lasers:
            laser["pos"] += laser["vel"] * DELTA_T
            laser["lifetime"] -= 1
            if laser["lifetime"] > 0 and 0 <= laser["pos"][0] <= self.deltaX and -self.deltaZ <= laser["pos"][1] <= 0:
                newLasers.append(laser)
        self.lasers = newLasers
        self.checkLaserCollisions()

    def checkLaserCollisions(self):
        newAsteroids = []
        for asteroid in self.asteroids:
            hit = False
            for laser in self.lasers:
                dist = np.linalg.norm(np.array(asteroid["pos"]) - laser["pos"])
                if dist < asteroid["radius"]:
                    hit = True
                    break
            if not hit:
                newAsteroids.append(asteroid)
        self.asteroids = newAsteroids

    def isOutsideBounds(self, object):
        return (object.positionVector[0] < 0 or object.positionVector[0] > self.deltaX or
                object.positionVector[1] > 0 or object.positionVector[1] < -self.deltaZ)

    def plotForces(self, filename, orientationFilename, object):
        # Similar to Ocean.plotForces, adapted for Pilot
        orient = np.array(object.orientationVector, dtype=float)
        force = np.array(object.forceVector, dtype=float)
        norm = np.sqrt(orient @ orient)
        local_x_axis = orient / norm
        local_z_axis = np.array([-local_x_axis[1], local_x_axis[0]])
        globalXCoords = []
        globalZCoords = []
        for x_local, z_local in zip(object.geometry.pointXCoords, object.geometry.pointZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis + object.positionVector
            globalXCoords.append(global_point[0])
            globalZCoords.append(global_point[1])
        globalColocationXCoords = []
        globalColocationZCoords = []
        for x_local, z_local in zip(object.geometry.colocationXCoords, object.geometry.colocationZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis + object.positionVector
            globalColocationXCoords.append(global_point[0])
            globalColocationZCoords.append(global_point[1])
        centroidX = np.mean(globalXCoords)
        centroidZ = np.mean(globalZCoords)
        apparentCurrentVector = -np.array(object.velocityVector)
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(min(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(centroidX, centroidZ,
                  object.forceVector[0] * 0.01, object.forceVector[1] * 0.01,
                  head_width=0.01, head_length=0.01, fc='pink', ec='pink', label='Force Vector')
        plt.arrow(centroidX, centroidZ - 0.1,
                  object.thrustForce[0] * 0.01, object.thrustForce[1] * 0.01,
                  head_width=0.01, head_length=0.01, fc='blue', ec='blue', label='Thrust Vector')
        plt.xlim([0, self.deltaX])
        plt.ylim([-self.deltaZ, 1])
        plt.title("Global Velocity Frame")
        plt.legend()
        plt.savefig(os.path.join("graphs", filename))
        plt.close()

    def createMovie(self, movieFilename, frameFilenames):
        with imageio.get_writer(os.path.join('graphs', movieFilename), mode='I', duration=0.1) as writer:
            for frame in frameFilenames:
                image = imageio.imread(frame)
                writer.append_data(image)
                os.remove(frame)

    def plotVectorTimeseries(self, filename, vectors, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), [vector[0] for vector in vectors], label='X', linestyle='--')
        plt.plot(np.arange(self.frameNumber), [vector[1] for vector in vectors], label='Z')
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.title(f"{title} vs time")
        plt.legend()
        plt.savefig(os.path.join("graphs", filename))
        plt.close()

    def plotTimeseries(self, filename, values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), values, label=title, linestyle='--')
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.title(f"{title} vs time")
        plt.legend()
        plt.savefig(os.path.join("graphs", filename))
        plt.close()

class Object:
    def __init__(self, geometryData, positionX, positionZ):
        self.geometryData = geometryData
        self.geometry = self.geometryData.geometry
        self.mass = geometryData.mass
        self.positionVector = np.array([positionX, positionZ], dtype=np.float64)
        self.velocityVector = np.array([0.0, 0.0], dtype=np.float64)
        self.accelerationVector = GRAVITY_VECTOR / self.mass
        self.orientationVector = np.array([0.0, 0.0], dtype=np.float64)
        self.forceVector = np.array([0.0, 0.0], dtype=np.float64)
        self.addedMass = 0.0

    def rotateRight(self):
        if self.geometryData.hasTrailingEdge:
            currentAngle = np.arctan2(self.orientationVector[1], self.orientationVector[0])
            newAngle = np.radians(np.degrees(currentAngle) + DELTA_ROTATION)
            self.orientationVector = np.array([np.cos(newAngle), np.sin(newAngle)], dtype=np.float64)

    def rotateLeft(self):
        if self.geometryData.hasTrailingEdge:
            currentAngle = np.arctan2(self.orientationVector[1], self.orientationVector[0])
            newAngle = np.radians(np.degrees(currentAngle) - DELTA_ROTATION)
            self.orientationVector = np.array([np.cos(newAngle), np.sin(newAngle)], dtype=np.float64)

    def thrustForward(self):
        self.thrustForce = self.orientationVector * self.mass * THRUST_STRENGTH  # 1 m/s^2 acceleration

    def thrustBackward(self):
        self.thrustForce = self.orientationVector * -self.mass * THRUST_STRENGTH
        
    def disableThrust(self):
        self.thrustForce = np.array([0, 0], dtype=np.float64) 
        
    def pointUp(self):
        self.orientationVector = np.array([0.0, 1.0], dtype=np.float64)
        
    def pointRight(self):
        self.orientationVector = np.array([1.0, 0.0], dtype=np.float64)

    def updatePosition(self):
        try:
            self.updateForce(self.velocityVector, self.accelerationVector)
            totalForceVector = self.forceVector + self.thrustForce
            self.accelerationVector = totalForceVector / (self.mass + self.addedMass)
            self.capAcceleration()
            self.velocityVector += self.accelerationVector * DELTA_T
            self.capVelocity()
            self.positionVector += self.velocityVector * DELTA_T
        except Exception as e:
            print(f"Error in updatePosition: {e}")
            raise

    def updateForce(self, velocityVector, accelerationVector):
        try:
            self.velocityVector = velocityVector.astype(np.float64)
            self.accelerationVector = accelerationVector.astype(np.float64)
            self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, velocityVector, accelerationVector)
            accelerationMagnitude = np.sqrt(self.accelerationVector[0]**2 + self.accelerationVector[1]**2)
            forceMagnitude = np.sqrt(self.forceVector[0]**2 + self.forceVector[1]**2)
            self.addedMass = forceMagnitude / accelerationMagnitude if accelerationMagnitude > 0 else 0.0
        except Exception as e:
            print(f"Error in updateForce: {e}")
            raise

    def capAcceleration(self):
        self.accelerationVector = np.clip(self.accelerationVector, -MAX_ACCELERATION, MAX_ACCELERATION)

    def capVelocity(self):
        self.velocityVector = np.clip(self.velocityVector, -MAX_VELOCITY, MAX_VELOCITY)