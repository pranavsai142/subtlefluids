from Geometry import Geometry, Circle, Foil
from tunnel import Tunnel
from renderer import Renderer

import numpy as np

# HYDROFOILER
# Main Menu Options
# The Shop
# Tunnel
# Ocean
# Pilot

# New driver


# End New driver

# Construct the object
foil = Foil("0012", 0.203, 300)
foil.geometry.plotGeometry("naca12_foil_geometry.png")
foil.geometry.plotNormals("naca12_foil_normals.png")

circle = Circle(1, 200)
circle.geometry.plotGeometry("circle_geometry.png")
circle.geometry.plotNormals("circle_normals.png")

# Constants for flow generation
T = 100
OMEGA = 2 * np.pi / T
# OMEGA = 0
WIND_SPEED = 10
ALPHA = np.radians(0)
U_0 = WIND_SPEED  # Assuming U_0 should be WIND_SPEED based on context

def generateUnsteadyFlow(frame):
    sinOmegaT = np.sin(OMEGA * frame)
    cosOmegaT = np.cos(OMEGA * frame)
    velocityX = WIND_SPEED * sinOmegaT * np.cos(ALPHA)
    velocityZ = WIND_SPEED * sinOmegaT * np.sin(ALPHA)
    accelerationX = U_0 * OMEGA * cosOmegaT * np.cos(ALPHA)
    accelerationZ = U_0 * OMEGA * cosOmegaT * np.sin(ALPHA)
#     velocityX = 1  # Using the hardcoded values as per your original function
#     velocityZ = 0
    velocity = [-velocityX, -velocityZ]
#     accelerationX = 0
#     accelerationZ = 0
    acceleration = [-accelerationX, -accelerationZ]
#     print(acceleration)
#     quit()
    return velocity, acceleration
    
def generateSteadyFlow(frame):
    velocityX = WIND_SPEED * np.cos(ALPHA)
    velocityZ = WIND_SPEED * np.sin(ALPHA)
#     velocityX = 1  # Using the hardcoded values as per your original function
#     velocityZ = 0
    velocity = [-velocityX, -velocityZ]
#     accelerationX = 0
#     accelerationZ = 0
    acceleration = [0, 0]
    return velocity, acceleration

# Initialize the Tunnel
tunnel = Tunnel()

# Initialize the Renderer for Tunnel
renderer = Renderer(windowWidth=1000, windowHeight=800, environment="tunnel")

# Add an object to the Tunnel (assuming 'foil' is defined)
tunnelObject = tunnel.addObject(circle)

# Evolve the motion of the object
frame = 0
while renderer.isRunning():
    tunnelVelocity, tunnelAcceleration = generateUnsteadyFlow(frame)
    tunnel.advanceTime(tunnelVelocity, tunnelAcceleration)
    renderer.render(tunnel)
#     tunnel.printEnvironment()
    frame += 1

renderer.quit()


# Initialize the Ocean
ocean = Ocean(100, 100)
oceanObject = ocean.addObject(circle)  # Assuming 'foil' is defined

# Initialize the Renderer for Ocean
renderer = Renderer(windowWidth=1000, windowHeight=800, deltaX=ocean.deltaX, deltaZ=ocean.deltaZ, environment="ocean")

# Evolve the motion of the object
leftKeyPressed = True
while renderer.isRunning():
    # if leftKeyPressed:
    #     oceanObject.rotateLeft()
    ocean.advanceTime()
    renderer.render(ocean)
#     ocean.printEnvironment()

renderer.quit()

