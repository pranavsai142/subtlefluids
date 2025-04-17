from Geometry import Geometry, Circle, Foil

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
        self.velocityVector = [0, 1]
        self.accelerationVector = [0, 0]
#         Start pointing straight down
        self.orientationVector = [0, -1]
        self.angleOfAttack = 0
        
    def rotateLeft(self):
        self.orientationVector = [0, -1]
        
    def updateAngleOfAttack(self):
        self.angleOfAttack += 1
        
    def updatePosition(self):
        forceVector = self.geometry.computeForceFromFlow(self.orientationVector, self.velocityVector, self.accelerationVector)
        print("Force Vector from Flow", forceVector)
        self.positionVector[1] -= 0.1
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