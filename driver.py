from Geometry import Geometry, Circle, Foil
from Ocean import Ocean



# Construct the object
foil = Foil("0012", 0.203, 20)
foil.geometry.plotGeometry("naca12_foil_geometry.png")


circle = Circle(1, 20)
circle.geometry.plotGeometry("circle_geometry.png")

# Add the object to environment
ocean = Ocean(5, 20)
circleObject = ocean.addObject(circle)

# Evolve the motion of the object
leftKeyPressed = True
while(True):
    if(leftKeyPressed):
        circleObject.rotateLeft()
    ocean.advanceTime()
    ocean.printEnvironment()
# circle.generateInfluenceMatrices()