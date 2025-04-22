from Geometry import Geometry, Circle, Foil
import numpy as np
import matplotlib.pyplot as plt
import os

class Ocean:
    def __init__(self, deltaX, deltaZ):
        print("Initializing Ocean")
        self.deltaX = deltaX
        self.deltaZ = deltaZ
        self.objects = []
        
    def addObject(self, geometryData):
        object = Object(geometryData, self.deltaX/2, 0)
        self.objects.append(object)
        return object
        
    def advanceTime(self):
        for object in self.objects:
            object.updatePosition()
            if(self.isOutsideBounds(object)):
                print("CRASH!")
                quit()
            
    def isOutsideBounds(self, object):
        if(object.positionVector[0] < 0 
            or object.positionVector[0] > self.deltaX 
            or object.positionVector[1] > 0 
            or object.positionVector[1] < (-self.deltaZ)):
            return True
            
    def printEnvironment(self):
        for object in self.objects:
            print("object positionX, positionZ:", object.positionVector[0], object.positionVector[1])
        
class Object:
    def __init__(self, geometryData, positionX, positionZ):
        self.geometryData = geometryData
        self.geometry = self.geometryData.geometry
        self.mass = 0
        self.positionVector = [positionX, positionZ]
        self.velocityVector = [-0.0, -1]
        self.accelerationVector = [0, 9.8]
#         Start pointing straight down
        self.orientationVector = [0, -1]
        self.angleOfAttack = 0
        self.forceVector = [0,0]
        
    def rotateLeft(self):
        self.orientationVector = [0, -1]
        
    def updateAngleOfAttack(self):
        self.angleOfAttack += 1
        
    def updatePosition(self):
        self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, self.velocityVector, self.accelerationVector)
        self.plotForces("global_forces.png")
        

#         print("Force Vector from Flow", forceVector)
        self.positionVector[1] -= 0.1
        quit()
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

    def plotForces(self, filename):
        # Convert orientation and force to NumPy arrays
        orient = np.array(self.orientationVector, dtype=float)
        force = np.array(self.forceVector, dtype=float)
        
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
        for x_local, z_local in zip(self.geometry.pointXCoords, self.geometry.pointZCoords):
            # Local point = [x_local, z_local]
            # Global point = x_local * [-1, 0] + z_local * [0, 1] mapped to global basis
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis
            globalXCoords.append(global_point[0])
            globalZCoords.append(global_point[1])
        
        globalColocationXCoords = []
        globalColocationZCoords = []
        for x_local, z_local in zip(self.geometry.colocationXCoords, self.geometry.colocationZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis
            globalColocationXCoords.append(global_point[0])
            globalColocationZCoords.append(global_point[1])
        
        # Transform forceVector to global coordinates
        # Local force: [f_along, f_perp] along [-1, 0] and [0, 1]
        # Global force = f_along * (-local_x_axis) + f_perp * local_z_axis
        globalForce = force[0] * (-local_x_axis) + force[1] * local_z_axis
        
        # Compute centroid for force arrow (average of points)
        centroidX = np.mean(globalXCoords)
        centroidZ = np.mean(globalZCoords)
        
        apparentCurrentVector = -np.array(self.velocityVector)
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(0, min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(-.1, min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.arrow(.1, min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.arrow(.2, min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
                  
        plt.arrow(centroidX, centroidZ,
                self.forceVector[0] * 0.01, self.forceVector[1] * 0.01,
                head_width=0.01, head_length=0.01, fc='pink', ec='pink', label='Force Vector')
                  
        plt.title("Global Velocity Frame")
        plt.legend()
        plt.savefig(os.path.join('graphs', filename))
        plt.close()
