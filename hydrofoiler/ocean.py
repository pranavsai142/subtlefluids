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

class Ocean:
    def __init__(self, deltaX, deltaZ):
        print("Initializing Ocean")
        self.deltaX = deltaX
        self.deltaZ = deltaZ
        self.objects = []
        self.frameNumber = 0
        self.frameFilenames = []
        self.orientationFrameFilenames = []
        self.objectFrameFilenames = []
        self.forceVectors = []
        self.positionVectors = []
        self.velocityVectors = []
        self.accelerationVectors = []
        self.addedMasses = []
        self.pathHistory = []  # For minimap
        self.running = True
        self.maxHistoryLength = 100  # Cap history lengths

    def addObject(self, geometryData):
        object = Object(geometryData, self.deltaX/2, 0)
        object.pointDown()
        self.objects.append(object)
        return object

    def handleKey(self, event):
        if event.key == pygame.K_LEFT:
            for obj in self.objects:
                obj.rotateLeft()
        elif event.key == pygame.K_RIGHT:
            for obj in self.objects:
                obj.rotateRight()
                
    def handleKeys(self, keys):
        if keys[pygame.K_LEFT]:
            for obj in self.objects:
                obj.rotateLeft()
        if keys[pygame.K_RIGHT]:
            for obj in self.objects:
                obj.rotateRight()

    def updateSize(self, deltaX, deltaZ):
        self.deltaX = max(10, deltaX)  # Minimum size to prevent issues
        self.deltaZ = max(10, deltaZ)

    def advanceTime(self):
        for object in self.objects:
            object.updatePosition()
            self.pathHistory.append(object.positionVector.copy())
            if len(self.pathHistory) > self.maxHistoryLength:
                self.pathHistory.pop(0)
            if self.isOutsideBounds(object) or self.frameNumber >= MAX_FRAMES:
                print("CRASH!", object.positionVector, self.isOutsideBounds(object))
                if PLOT_FRAMES:
                    self.createMovie("ocean_global_forces.gif", self.frameFilenames)
                    self.createMovie("ocean_local_forces.gif", self.objectFrameFilenames)
                    self.createMovie("ocean_local_orientations.gif", self.orientationFrameFilenames)
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_forces_timeseries.png"), self.forceVectors, "force")
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_position_timeseries.png"), self.positionVectors, "position")
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_velocity_timeseries.png"), self.velocityVectors, "velocity")
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_acceleration_timeseries.png"), self.accelerationVectors, "acceleration")
                    self.plotTimeseries(os.path.join("graphs", "ocean_added_mass_timeseries.png"), self.addedMasses, "added mass")
                self.running = False
            else:
                if PLOT_FRAMES:
                    frameFilename = os.path.join("graphs", f"ocean_global_forces_{self.frameNumber}.png")
                    orientationFrameFilename = os.path.join("graphs", f"ocean_local_orientation_{self.frameNumber}.png")
                    self.plotForces(frameFilename, orientationFrameFilename, object)
                    self.frameFilenames.append(frameFilename)
                    self.orientationFrameFilenames.append(orientationFrameFilename)
                    objectFrameFilename = os.path.join("graphs", f"ocean_local_forces_{self.frameNumber}.png")
                    object.geometry.plotForces(objectFrameFilename, object.geometry.localVelocityVector,
                                              object.geometry.tangentialTotalVelocity, object.geometry.localForceVector)
                    self.objectFrameFilenames.append(objectFrameFilename)
                    # Cap frame filenames
                    if len(self.frameFilenames) > self.maxHistoryLength:
                        self.frameFilenames.pop(0)
                    if len(self.orientationFrameFilenames) > self.maxHistoryLength:
                        self.orientationFrameFilenames.pop(0)
                    if len(self.objectFrameFilenames) > self.maxHistoryLength:
                        self.objectFrameFilenames.pop(0)
                    self.forceVectors.append(object.forceVector)
                    self.positionVectors.append(copy.copy(object.positionVector))
                    self.velocityVectors.append(copy.copy(object.velocityVector))
                    self.accelerationVectors.append(copy.copy(object.accelerationVector))
                    self.addedMasses.append(copy.copy(object.addedMass))
                    # Cap vector lists
                    if len(self.forceVectors) > self.maxHistoryLength:
                        self.forceVectors.pop(0)
                    if len(self.positionVectors) > self.maxHistoryLength:
                        self.positionVectors.pop(0)
                    if len(self.velocityVectors) > self.maxHistoryLength:
                        self.velocityVectors.pop(0)
                    if len(self.accelerationVectors) > self.maxHistoryLength:
                        self.accelerationVectors.pop(0)
                    if len(self.addedMasses) > self.maxHistoryLength:
                        self.addedMasses.pop(0)
                self.frameNumber += 1
                
    def cleanup(self):
        # Clear lists to free memory, but preserve objects (geometry)
        self.pathHistory.clear()
        self.frameFilenames.clear()
        self.orientationFrameFilenames.clear()
        self.objectFrameFilenames.clear()
        self.forceVectors.clear()
        self.positionVectors.clear()
        self.velocityVectors.clear()
        self.accelerationVectors.clear()
        self.addedMasses.clear()
        # Close any open Matplotlib figures
        plt.close('all')


    def isOutsideBounds(self, object):
        return (object.positionVector[0] < 0 or object.positionVector[0] > self.deltaX or
                object.positionVector[1] > 0 or object.positionVector[1] < -self.deltaZ)

    def plotForces(self, filename, orientationFilename, object):
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
        plt.arrow(max(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.arrow(centroidX, centroidZ,
                  object.forceVector[0] * 0.01, object.forceVector[1] * 0.01,
                  head_width=0.01, head_length=0.01, fc='pink', ec='pink', label='Force Vector')
        plt.xlim([0, self.deltaX])
        plt.ylim([-self.deltaZ, 1])
        plt.title("Global Velocity Frame")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(min(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(max(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.title("Orientation Frame")
        plt.legend()
        plt.savefig(orientationFilename)
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
        plt.savefig(filename)
        plt.close()

    def plotTimeseries(self, filename, values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), values, label=title, linestyle='--')
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.title(f"{title} vs time")
        plt.legend()
        plt.savefig(filename)
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

    def pointDown(self):
        self.orientationVector = np.array([0.0, -1.0], dtype=np.float64)

    def capAcceleration(self):
        self.accelerationVector = np.clip(self.accelerationVector, -MAX_ACCELERATION, MAX_ACCELERATION)

    def capVelocity(self):
        self.velocityVector = np.clip(self.velocityVector, -MAX_VELOCITY, MAX_VELOCITY)

    def updatePosition(self):
        self.updateForce(self.velocityVector, self.accelerationVector)
        totalForceVector = self.forceVector + ((self.mass + self.addedMass) * GRAVITY_VECTOR)
        totalForceVector = self.forceVector + ((self.mass) * GRAVITY_VECTOR)
        self.accelerationVector = totalForceVector / (self.mass + self.addedMass)
        self.capAcceleration()
        self.velocityVector += self.accelerationVector * DELTA_T
        self.capVelocity()
        self.positionVector += self.velocityVector * DELTA_T

    def updateForce(self, velocityVector, accelerationVector):
        self.velocityVector = velocityVector
        self.accelerationVector = accelerationVector
        self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, velocityVector, accelerationVector)
        accelerationMagnitude = np.sqrt(self.accelerationVector[0]**2 + self.accelerationVector[1]**2)
        forceMagnitude = np.sqrt(self.forceVector[0]**2 + self.forceVector[1]**2)
        self.addedMass = forceMagnitude / accelerationMagnitude if accelerationMagnitude > 0 else 0.0