from Geometry import Geometry, Circle, Foil
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

MAX_FRAMES = 100
class Ocean:
    def __init__(self, deltaX, deltaZ):
        print("Initializing Ocean")
        self.deltaX = deltaX
        self.deltaZ = deltaZ
        self.objects = []
        self.frameNumber = 0
        self.frameFilenames = []
        self.objectFrameFilenames = []
        self.forceVectors = []
        self.positionVectors = []
        self.velocityVectors = []
        self.accelerationVectors = []
        
    def addObject(self, geometryData):
        object = Object(geometryData, self.deltaX/2, 0)
        self.objects.append(object)
        return object
        
    def advanceTime(self):
        for object in self.objects:
            object.updatePosition()
            if(self.isOutsideBounds(object) or self.frameNumber >= MAX_FRAMES):
                print("CRASH!", object.positionVector, self.isOutsideBounds(object))
                self.createMovie("global_forces.gif", self.frameFilenames)
                self.createMovie("local_forces.gif", self.objectFrameFilenames)
                self.plotTimeseries(os.path.join("graphs", "global_forces_timeseries.png"), self.forceVectors, "force")
                self.plotTimeseries(os.path.join("graphs", "global_position_timeseries.png"), self.positionVectors, "position")
                self.plotTimeseries(os.path.join("graphs", "global_velocity_timeseries.png"), self.velocityVectors, "velocity")
                self.plotTimeseries(os.path.join("graphs", "global_acceleration_timeseries.png"), self.accelerationVectors, "acceleration")
                
                quit()
            else:
                frameFilename = os.path.join("graphs", "global_forces_" + str(self.frameNumber) + ".png")
                self.plotForces(frameFilename, object)
                self.frameFilenames.append(frameFilename)
                self.forceVectors.append(object.forceVector)
                self.positionVectors.append(object.positionVector)
                self.velocityVectors.append(object.velocityVector)
                self.accelerationVectors.append(object.accelerationVector)
            
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
            
    def plotForces(self, filename, object):
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
        
    def createMovie(self, movieFilename, frameFilenames):
        with imageio.get_writer(os.path.join('graphs', movieFilename), mode='I', duration=0.1) as writer:
            for frame in frameFilenames:
                image = imageio.imread(frame)
                writer.append_data(image)
                os.remove(frame)

    def plotTimeseries(self, filename, vectors, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), [vector[0] for vector in vectors], label='X', linestyle='--')
        plt.plot(np.arange(self.frameNumber), [vector[1] for vector in vectors], label='Z')
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.title(title + " vs time")
        plt.legend()
#         plt.ylim([-30, 30])     
    #     plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
        plt.savefig(filename)
        plt.close()

        
GRAVITY = .0981
DELTA_T = 0.01
MAX_VELOCITY = 1000000
MAX_ACCELERATION = 1000000
class Object:
    def __init__(self, geometryData, positionX, positionZ):
        self.geometryData = geometryData
        self.geometry = self.geometryData.geometry
        self.mass = 300
        self.positionVector = [positionX, positionZ]
        print(self.positionVector)
#         quit()
        self.velocityVector = [0, 0]
        self.accelerationVector = [0.0, 0]
#         Start pointing straight down
        self.orientationVector = [0, -1]
        self.angleOfAttack = 0
        self.forceVector = [0,0]
        
    def rotateLeft(self):
        self.orientationVector = [0, -1]
        
    def updateAngleOfAttack(self):
        self.angleOfAttack += 1
        
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
        
    def updatePosition(self):
#         self.velocityVector[0] += 0.001
#         self.velocityVector[1] -= 0.01
#         self.accelerationVector = [0.0, -3]
        self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, self.velocityVector, self.accelerationVector)
        print("global force vector", self.forceVector)
#         self.velocityVector[1] += -1
        gravityVector = np.array([0, -GRAVITY])
#         
# #         # Compute total force: external force + gravity
        totalForceVector = self.forceVector + (self.mass * gravityVector)
        totalForceVector = (self.mass * gravityVector)
        
        # Compute acceleration: a = F/m
        self.accelerationVector = totalForceVector / self.mass
        self.capAcceleration()
# #         print(self.accelerationVector)
#         # Update velocity: v(t + Δt) = v(t) + a(t) * Δt
        self.velocityVector += self.accelerationVector * DELTA_T
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

