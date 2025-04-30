import pygame
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import copy
import math
from geometry import Geometry, Circle, Foil

MAX_FRAMES = 10000
PLOT_FRAMES = False
DELTA_ROTATION = 1

class Tunnel:
    def __init__(self):
        print("Initializing Tunnel")
        self.objects = []
        self.frameNumber = 0
        self.frameFilenames = []
        self.orientationFrameFilenames = []
        self.objectFrameFilenames = []
        self.forceVectors = []
        self.maxHistoryLength = 100
        self.running = True

    def addObject(self, geometryData):
        object = Object(geometryData, 0, 0)
        object.pointLeft()
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

    def advanceTime(self, velocity, acceleration, alphaDeg, unsteady, T):
        alphaRad = math.radians(alphaDeg)
        velocityX = velocity * math.cos(alphaRad)
        velocityZ = velocity * math.sin(alphaRad)
        if unsteady:
            t = (self.frameNumber % T) / T
            accelerationX = acceleration * math.cos(alphaRad) * math.sin(2 * math.pi * t)
            accelerationZ = acceleration * math.sin(alphaRad) * math.sin(2 * math.pi * t)
        else:
            accelerationX = 0
            accelerationZ = 0
        velocityVector = [-velocityX, -velocityZ]
        accelerationVector = [-accelerationX, -accelerationZ]
        for obj in self.objects:
            obj.updateForce(velocityVector, accelerationVector)
            if PLOT_FRAMES:
                self.forceVectors.append(obj.forceVector)
                if len(self.forceVectors) > self.maxHistoryLength:
                    self.forceVectors.pop(0)
                frameFilename = os.path.join("graphs", f"tunnel_global_forces_{self.frameNumber}.png")
                orientationFrameFilename = os.path.join("graphs", f"tunnel_local_orientation_{self.frameNumber}.png")
                self.plotForces(frameFilename, orientationFrameFilename, obj)
                self.frameFilenames.append(frameFilename)
                self.orientationFrameFilenames.append(orientationFrameFilename)
                objectFrameFilename = os.path.join("graphs", f"tunnel_local_forces_{self.frameNumber}.png")
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
            
    def cleanup(self):
        self.frameFilenames.clear()
        self.orientationFrameFilenames.clear()
        self.objectFrameFilenames.clear()
        self.forceVectors.clear()
        plt.close('all')

    def printEnvironment(self):
        for object in self.objects:
            print("In The Tunnel, Velocity Acceleration", object.velocityVector, object.accelerationVector)

    def plotForces(self, filename, orientationFilename, object):
        orient = np.array(object.orientationVector, dtype=float)
        force = np.array(object.forceVector, dtype=float)
        norm = np.sqrt(orient @ orient)
        local_x_axis = orient / norm
        local_z_axis = np.array([-local_x_axis[1], local_x_axis[0]])
        globalXCoords = []
        globalZCoords = []
        for x_local, z_local in zip(object.geometry.pointXCoords, object.geometry.pointZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis
            global_point += object.positionVector
            globalXCoords.append(global_point[0])
            globalZCoords.append(global_point[1])
        globalColocationXCoords = []
        globalColocationZCoords = []
        for x_local, z_local in zip(object.geometry.colocationXCoords, object.geometry.colocationZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis
            global_point += object.positionVector
            globalColocationXCoords.append(global_point[0])
            globalColocationZCoords.append(global_point[1])
        centroidX = np.mean(globalXCoords)
        centroidZ = np.mean(globalZCoords)
        apparentCurrentVector = -np.array(object.velocityVector)
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(min(globalXCoords) - 0.1, 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(max(globalXCoords) - 0.1, -0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.arrow(centroidX, centroidZ,
                  object.forceVector[0] * 0.01, object.forceVector[1] * 0.01,
                  head_width=0.01, head_length=0.01, fc='pink', ec='pink', label='Force Vector')
        plt.title("Global Velocity Frame")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(min(globalXCoords) - 0.1, 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(max(globalXCoords) - 0.1, -0.1,
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
        self.positionVector = [positionX, positionZ]
        self.velocityVector = [0, 0]
        self.accelerationVector = [0, 0]
        self.orientationVector = [0, 0]
        self.forceVector = [0, 0]
        self.addedMass = 0

    def rotateRight(self):
        if self.geometryData.hasTrailingEdge:
            currentAngle = math.atan2(self.orientationVector[1], self.orientationVector[0])
            newAngle = math.radians(math.degrees(currentAngle) + DELTA_ROTATION)
            self.orientationVector = [math.cos(newAngle), math.sin(newAngle)]

    def rotateLeft(self):
        if self.geometryData.hasTrailingEdge:
            currentAngle = math.atan2(self.orientationVector[1], self.orientationVector[0])
            newAngle = math.radians(math.degrees(currentAngle) - DELTA_ROTATION)
            self.orientationVector = [math.cos(newAngle), math.sin(newAngle)]

    def pointLeft(self):
        self.orientationVector = [-1, 0]

    def updateForce(self, velocityVector, accelerationVector):
        self.velocityVector = velocityVector
        self.accelerationVector = accelerationVector
        self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, velocityVector, accelerationVector)
        accelerationMagnitude = np.sqrt(self.accelerationVector[0]**2 + self.accelerationVector[1]**2)
        forceMagnitude = np.sqrt(self.forceVector[0]**2 + self.forceVector[1]**2)
        self.addedMass = forceMagnitude / accelerationMagnitude if accelerationMagnitude > 0 else 0