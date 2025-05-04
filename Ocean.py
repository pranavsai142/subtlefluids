from Geometry import Geometry, Circle, Foil
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import math
import copy

MAX_FRAMES = 1000
PLOT_FRAMES = True
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
        self.running = True
        
    def addObject(self, geometryData):
        object = Object(geometryData, self.deltaX/2, 0)
        object.pointDown()
        self.objects.append(object)
        return object
        
    def advanceTime(self):
        for object in self.objects:
            object.updatePosition()
            if(self.isOutsideBounds(object) or self.frameNumber >= MAX_FRAMES):
                print("CRASH!", object.positionVector, self.isOutsideBounds(object))
                if(PLOT_FRAMES):
                    self.createMovie("global_forces.gif", self.frameFilenames)
                    self.createMovie("local_forces.gif", self.objectFrameFilenames)
                    self.createMovie("local_orientations.gif", self.orientationFrameFilenames)
                self.plotVectorTimeseries(os.path.join("graphs", "global_forces_timeseries.png"), self.forceVectors, "force")
                self.plotVectorTimeseries(os.path.join("graphs", "global_position_timeseries.png"), self.positionVectors, "position")
                self.plotVectorTimeseries(os.path.join("graphs", "global_velocity_timeseries.png"), self.velocityVectors, "velocity")
                self.plotVectorTimeseries(os.path.join("graphs", "global_acceleration_timeseries.png"), self.accelerationVectors, "acceleration")
                self.plotTimeseries(os.path.join("graphs", "global_added_mass_timeseries.png"), self.addedMasses, "added mass")
                self.running = False
            else:
                self.forceVectors.append(object.forceVector)
                self.positionVectors.append(copy.copy(object.positionVector))
                self.velocityVectors.append(copy.copy(object.velocityVector))
                self.accelerationVectors.append(copy.copy(object.accelerationVector))
                self.addedMasses.append(copy.copy(object.addedMass))
                
                if(PLOT_FRAMES):
                    frameFilename = os.path.join("graphs", "global_forces_" + str(self.frameNumber) + ".png")
                    orientationFrameFilename = os.path.join("graphs", "global_orientation_" + str(self.frameNumber) + ".png")
                    self.plotForces(frameFilename, orientationFrameFilename, object)
                    self.frameFilenames.append(frameFilename)
                    self.orientationFrameFilenames.append(orientationFrameFilename)
                    objectFrameFilename = os.path.join("graphs", "local_forces_" + str(self.frameNumber) + ".png")
                    object.geometry.plotForces(objectFrameFilename, object.geometry.localVelocityVector, object.geometry.tangentialTotalVelocity, object.geometry.localForceVector)
                    self.objectFrameFilenames.append(objectFrameFilename)
            
                self.frameNumber += 1
            
    def isOutsideBounds(self, object):
        if(object.positionVector[0] < 0 
            or object.positionVector[0] > self.deltaX 
            or object.positionVector[1] > 0 
            or object.positionVector[1] < (-self.deltaZ)):
            return True
            
    def printEnvironment(self):
        for object in self.objects:
            print("object positionX, positionZ:", object.positionVector[0], object.positionVector[1])
            
    def plotForces(self, filename, orientationFilename, object):
        # Convert orientation and force to NumPy arrays
        orient = np.array(object.orientationVector, dtype=float)
        force = np.array(object.forceVector, dtype=float)
        
        # Normalize orientation vector to get local x-axis in global coordinates
        norm = np.sqrt(orient @ orient)
        local_x_axis = orient / norm  # Global direction of local x-axis ([-1, 0] in local)
        
        # Compute local z-axis in global coordinates (perpendicular, 90° counterclockwise)
        local_z_axis = np.array([-local_x_axis[1], local_x_axis[0]])
        
        # Transform local coordinates to global coordinates
        # Local basis: x-axis = [-1, 0], z-axis = [0, 1]
        # Global coords = x_local * (-local_x_axis) + z_local * local_z_axis
        globalXCoords = []
        globalZCoords = []
        for x_local, z_local in zip(object.geometry.pointXCoords, object.geometry.pointZCoords):
            # Local point = [x_local, z_local]
            # Global point = x_local * [-1, 0] + z_local * [0, 1] mapped to global basis
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
        
        # Transform forceVector to global coordinates
        # Local force: [f_along, f_perp] along [-1, 0] and [0, 1]
        # Global force = f_along * (-local_x_axis) + f_perp * local_z_axis
        globalForce = force[0] * (-local_x_axis) + force[1] * local_z_axis
        
        # Compute centroid for force arrow (average of points)
        centroidX = np.mean(globalXCoords)
        centroidZ = np.mean(globalZCoords)
        
        apparentCurrentVector = -np.array(object.velocityVector)
        plt.grid(True)
#         plt.axis('equal')
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
        
#         plt.xlim([0, self.deltaX])
#         plt.ylim([-self.deltaZ, 1])          
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
        plt.xlabel("Time (frames)")
        plt.title(title + " vs time")
        plt.legend()
#         plt.ylim([-30, 30])     
#         plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
        plt.savefig(filename)
        plt.close()
        
    def plotTimeseries(self, filename, values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), values, label=title, linestyle='--')
        plt.grid(True)
        plt.xlabel("Time (frames)")
        plt.title(title + " vs time")
        plt.legend()
        plt.ylim([0, 4000])     
#         plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
        plt.savefig(filename)
        plt.close()


class Tunnel:
    def __init__(self):
        print("Initializing Tunnel")
        self.objects = []
        self.frameNumber = 0
        self.frameFilenames = []
        self.orientationFrameFilenames = []
        self.objectFrameFilenames = []
        self.forceVectors = []
        self.velocityVectors = []
        self.accelerationVectors = []
        self.addedMasses = []
        self.running = True
        
    def addObject(self, geometryData):
        object = Object(geometryData, 0, 0)
        object.pointLeft()
        self.objects.append(object)
        return object
        
    def advanceTime(self, velocity, acceleration):
        for object in self.objects:
            if(self.frameNumber >= MAX_FRAMES):
                print("DONE! Tunnel Generating Plots")
                if(PLOT_FRAMES):
                    self.createMovie("tunnel_global_forces.gif", self.frameFilenames)
                    self.createMovie("tunnel_local_forces.gif", self.objectFrameFilenames)
                    self.createMovie("tunnel_local_orientations.gif", self.orientationFrameFilenames)
                self.plotVectorTimeseries(os.path.join("graphs", "tunnel_forces_timeseries.png"), self.forceVectors, "force")
                self.plotVectorTimeseries(os.path.join("graphs", "tunnel_velocity_timeseries.png"), self.velocityVectors, "velocity")
                self.plotVectorTimeseries(os.path.join("graphs", "tunnel_acceleration_timeseries.png"), self.accelerationVectors, "acceleration")
                self.plotTimeseries(os.path.join("graphs", "tunnel_added_mass_timeseries.png"), self.addedMasses, "added mass")
                self.running = False
            else:
                object.updateForce(velocity, acceleration)
                self.forceVectors.append(object.forceVector)
                self.velocityVectors.append(copy.copy(object.velocityVector))
                self.accelerationVectors.append(copy.copy(object.accelerationVector))
                self.addedMasses.append(copy.copy(object.addedMass))
            
                if(PLOT_FRAMES):
                    frameFilename = os.path.join("graphs", "tunnel_global_forces_" + str(self.frameNumber) + ".png")
                    orientationFrameFilename = os.path.join("graphs", "tunnel_local_orientation_" + str(self.frameNumber) + ".png")
                    self.plotForces(frameFilename, orientationFrameFilename, object)
                    self.frameFilenames.append(frameFilename)
                    self.orientationFrameFilenames.append(orientationFrameFilename)
                    objectFrameFilename = os.path.join("graphs", "tunnel_local_forces_" + str(self.frameNumber) + ".png")
                    object.geometry.plotForces(objectFrameFilename, object.geometry.localVelocityVector, object.geometry.tangentialTotalVelocity, object.geometry.localForceVector)
                    self.objectFrameFilenames.append(objectFrameFilename)
        
                self.frameNumber += 1
            
            
    def printEnvironment(self):
        for object in self.objects:
            print("In The Tunnel, Velocity Acceleration", object.velocityVector, object.accelerationVector)
            
    def plotForces(self, filename, orientationFilename, object):
        # Convert orientation and force to NumPy arrays
        orient = np.array(object.orientationVector, dtype=float)
        force = np.array(object.forceVector, dtype=float)
        
        # Normalize orientation vector to get local x-axis in global coordinates
        norm = np.sqrt(orient @ orient)
        local_x_axis = orient / norm  # Global direction of local x-axis ([-1, 0] in local)
        
        # Compute local z-axis in global coordinates (perpendicular, 90° counterclockwise)
        local_z_axis = np.array([-local_x_axis[1], local_x_axis[0]])
        
        # Transform local coordinates to global coordinates
        # Local basis: x-axis = [-1, 0], z-axis = [0, 1]
        # Global coords = x_local * (-local_x_axis) + z_local * local_z_axis
        globalXCoords = []
        globalZCoords = []
        for x_local, z_local in zip(object.geometry.pointXCoords, object.geometry.pointZCoords):
            # Local point = [x_local, z_local]
            # Global point = x_local * [-1, 0] + z_local * [0, 1] mapped to global basis
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
        
        # Transform forceVector to global coordinates
        # Local force: [f_along, f_perp] along [-1, 0] and [0, 1]
        # Global force = f_along * (-local_x_axis) + f_perp * local_z_axis
        globalForce = force[0] * (-local_x_axis) + force[1] * local_z_axis
        
        # Compute centroid for force arrow (average of points)
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
        plt.xlabel("Time (frames)")
        plt.title(title + " vs time")
        plt.legend()
#         plt.ylim([-30, 30])     
#         plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
        plt.savefig(filename)
        plt.close()
        
    def plotTimeseries(self, filename, values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), values, label=title, linestyle='--')
        plt.grid(True)
        plt.xlabel("Time (frames)")
        plt.title(title + " vs time")
        plt.legend()
#         plt.ylim([0, 4000])     
#         plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
        plt.savefig(filename)
        plt.close()

        
GRAVITY = 9.81
GRAVITY_VECTOR = np.array([0, -GRAVITY])
DELTA_T = 0.01
MAX_VELOCITY = 1000000
MAX_ACCELERATION = 1000000
DELTA_ROTATION = 2.0
class Object:
    def __init__(self, geometryData, positionX, positionZ):
        self.geometryData = geometryData
        self.geometry = self.geometryData.geometry
        self.mass = 100
        self.positionVector = [positionX, positionZ]
        print(self.positionVector)
#         quit()
        self.velocityVector = [0, 0]
        self.accelerationVector = GRAVITY_VECTOR / self.mass
#         Start pointing straight down
        self.orientationVector = [0, 0]
        self.forceVector = [0,0]
        
    def rotateRight(self):
        """
        Rotate the orientation vector counterclockwise by DELTA_ROTATION degrees.
        """
        # Convert current orientation to angle
        currentAngle = math.atan2(self.orientationVector[1], self.orientationVector[0])
        # Convert to degrees, add DELTA_ROTATION, and convert back to radians
        newAngle = math.radians(math.degrees(currentAngle) + DELTA_ROTATION)
        # Update orientation vector
        self.orientationVector = [math.cos(newAngle), math.sin(newAngle)]

    def rotateLeft(self):
        """
        Rotate the orientation vector clockwise by DELTA_ROTATION degrees.
        """
        # Convert current orientation to angle
        currentAngle = math.atan2(self.orientationVector[1], self.orientationVector[0])
        # Convert to degrees, subtract DELTA_ROTATION, and convert back to radians
        newAngle = math.radians(math.degrees(currentAngle) - DELTA_ROTATION)
        # Update orientation vector
        self.orientationVector = [math.cos(newAngle), math.sin(newAngle)]
        
    def pointDown(self):
        self.orientationVector = [0, -1]
        
    def pointLeft(self):
        self.orientationVector = [-1, 0]
    
    def pointRight(self):
        self.orientationVector = [1, 0]
        
    def pointUp(self):
        self.orientationVector = [0, 1]
        
    def capAcceleration(self):
        if(self.accelerationVector[0] > MAX_ACCELERATION):
            self.accelerationVector[0] = MAX_ACCELERATION
        elif(self.accelerationVector[0] < -MAX_ACCELERATION):
            self.accelerationVector[0] = -MAX_ACCELERATION
        if(self.accelerationVector[1] > MAX_ACCELERATION):
            self.accelerationVector[1] = MAX_ACCELERATION
        elif(self.accelerationVector[1] < -MAX_ACCELERATION):
            self.accelerationVector[1] = -MAX_ACCELERATION
    
    def capVelocity(self):
        if(self.velocityVector[0] > MAX_VELOCITY):
            self.velocityVector[0] = MAX_VELOCITY
        elif(self.velocityVector[0] < -MAX_VELOCITY):
            self.velocityVector[0] = -MAX_VELOCITY
        if(self.velocityVector[1] > MAX_VELOCITY):
            self.velocityVector[1] = MAX_VELOCITY
        elif(self.velocityVector[1] < -MAX_VELOCITY):
            self.velocityVector[1] = -MAX_VELOCITY
        
    def updateForce(self, velocityVector, accelerationVector):
        self.velocityVector = velocityVector
        self.accelerationVector = accelerationVector
        self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, velocityVector, accelerationVector)
        # Normalize the velocity vector
        normalVelocity = velocityVector / np.linalg.norm(velocityVector)

        # Force component along (parallel to) velocity
        force_along_velocity = np.dot(self.forceVector, normalVelocity) * normalVelocity

        # Force component perpendicular to velocity
        force_perpendicular = self.forceVector - force_along_velocity

        # Print results
#         print("Global force vector along velocity (without gravity):", np.dot(self.forceVector, normalVelocity))
#         print("Force vector along velocity (parallel component):", force_along_velocity)
#         print("Force vector perpendicular to velocity:", force_perpendicular)
#         self.velocityVector[1] += -1
        accelerationMagnitude = np.sqrt(self.accelerationVector[0]**2 + self.accelerationVector[1]**2)
        forceMagnitude = np.sqrt(self.forceVector[0]**2 + self.forceVector[1]**2)
        self.addedMass = forceMagnitude / accelerationMagnitude
#         print("added mass", self.addedMass)
        
    def updatePosition(self):
#         self.velocityVector[0] += 0.001
#         self.velocityVector[1] -= 0.01
#         self.orientationVector[0] += 0.01
#         self.orientationVector[1] -= 0.01
# These values represet the second frame of freefall
#         self.velocityVector = [-1.63270090e-06, -9.80439634e-02]
#         self.accelerationVector = [-1.63270090e-04, -9.80439634e+00]
#         self.velocityVector = [0, -1]
#         self.accelerationVector = [0, 0]

#         self.accelerationVector = GRAVITY_VECTOR / self.mass
        self.updateForce(self.velocityVector, self.accelerationVector)
# #         # Compute total force: external force + gravity
#         totalForceVector = self.forceVector + ((self.mass + self.addedMass) * GRAVITY_VECTOR)
        totalForceVector = self.forceVector + ((self.mass) * GRAVITY_VECTOR)
#         self.forceVector = totalForceVector
#         gravityForceVector = (self.mass * GRAVITY_VECTOR)
#         print("total force vector", totalForceVector)
#         totalForceVector = (self.mass * gravityVector)
        
        # Compute acceleration: a = F/m
        self.accelerationVector = (totalForceVector / (self.mass + self.addedMass))
#         self.accelerationVector = (totalForceVector / (self.mass))
#         print("accelerationVector", self.accelerationVector)
        self.capAcceleration()
# #         print(self.accelerationVector)
#         # Update velocity: v(t + Δt) = v(t) + a(t) * Δt
        self.velocityVector += self.accelerationVector * DELTA_T
#         print("velocityVector", self.velocityVector)
        self.capVelocity()
#         print(self.velocityVector)
        
#         # Update position: p(t + Δt) = p(t) + v(t) * Δt
        self.positionVector += np.array(self.velocityVector) * DELTA_T
        

#         print("Force Vector from Flow", forceVector)
#          Compute rhs, based on apparent curent, which is opposite of velocity. Initially 0
#           Solve influence matrices from geometry
#          Solve for perturbation phi

#           Compute time derivative of rhs, how apparent current or velocity is changing
#           The change in velocity is dependent on the calculation of phiT
#           This is a coupled problem
#           Instead, assome at t=0, the object does not feel any inertial effects of the fluid
#           All it experiences is the acceleration of gravity in a vacuum
#           This assumption allows us to compute the time derivative of velocity at t=0
#           Using this, the rhs, and phiT can be solved for,

#           Calculate the forces on the object from the tangential velocity/dynamic pressure
#           Add the forces/acceleration due to added mass to the gravitiational acceleration

#           Calculate the new velocity by multiplying acceleration times a timestep
#           Calculate the new position by multiplying velocity times a timestep
#          
#           object.solveForFlow(velocity)
#           inside Foil class, willl have logic to handle KJ condition

