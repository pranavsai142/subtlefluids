import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import imageio.v2 as imageio
from functions import legpts, funf1, dfunf1, nacaz
from tqdm import tqdm

# Ensure the graphs folder exists
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Constants
RHO = 1000
NACA_CHORD_LENGTH = 0.203
NACA_FOIL_RESOLUTION = 0.001
NUM_POINTS = 301
NUM_POINTS_IN_ELEMENT = 3
NUM_REFERENCE_ELEMENT_POINTS = 100
COLOCATION_SCALING_FACTOR_X = 0.995
COLOCATION_SCALING_FACTOR_Z = 0.95
NUM_NACA_INTERNAL_POINTS_X = NUM_POINTS + 1
NUM_NACA_INTERNAL_POINTS_Z = NUM_POINTS + 1
SCALE_NACA_INTERNAL_GRID_X = 2
SCALE_NACA_INTERNAL_GRID_Z = 1

NUM_CIRCLE_POINTS = 20
CIRCLE_RADIUS = 1
CIRCLE_COLOCATION_SCALING_FACTOR = 0.95
NUM_CIRCLE_INTERNAL_POINTS = NUM_CIRCLE_POINTS + 1
SCALE_CIRCLE_INTERNAL_GRID = 3

# Time-varying flow parameters
U_0 = 1
T = 1.0
OMEGA = 2 * np.pi / T
NUM_TIME_STEPS = 100
DT = T / NUM_TIME_STEPS
# DT = 0.1
TIME_STEPS = np.linspace(0, T, NUM_TIME_STEPS + 1)

# Angles of attack to test (in degrees)
ALPHA_DEGREES = [0, 5, 8, 10, 11, 14, -8]
# ALPHA_DEGREES = [0, 8]
# New constants for plot limits
LIFT_COEFF_MIN = -700.0
LIFT_COEFF_MAX = 700.0
DRAG_COEFF_MIN = -30.0
DRAG_COEFF_MAX = 30.0
PERTURBATION_PHI_MIN = -0.03
PERTURBATION_PHI_MAX = 0.03
ADDED_MASS_MIN = 0.0
ADDED_MASS_MAX = 20.0
WIND_VELOCITY_MIN = -U_0 * 1.1  # Allow for negative velocities
WIND_VELOCITY_MAX = U_0 * 1.1   # Slightly above max velocity for visibility

NUM_VORTEX_POINTS = 100
INTEGRATE_VORTEX=True

SOLVE_CIRCLE = True
GRAPH_MAPS = False

# Utility Functions
def generateReferenceElementData():
    coords, weights = legpts(NUM_REFERENCE_ELEMENT_POINTS)
    shapeFunctions = funf1(coords, NUM_POINTS_IN_ELEMENT)
    shapeDerivatives = dfunf1(coords, NUM_POINTS_IN_ELEMENT)
    return coords, weights, shapeFunctions, shapeDerivatives

def computeArclengthInterpolation(xCoords, zCoords, numPoints):
    cumulativeArclengths = np.zeros(len(xCoords))
    for i in range(1, len(xCoords)):
        cumulativeArclengths[i] = cumulativeArclengths[i-1] + np.sqrt((xCoords[i] - xCoords[i-1])**2 + (zCoords[i] - zCoords[i-1])**2)
    arclengthMax = cumulativeArclengths[-1]
    pointArclengths = np.linspace(0, arclengthMax, numPoints)
    interpolateX = interp1d(cumulativeArclengths, xCoords, kind='cubic',
                            bounds_error=False, fill_value=(xCoords[0], xCoords[-1]))
    interpolateZ = interp1d(cumulativeArclengths, zCoords, kind='cubic',
                            bounds_error=False, fill_value=(zCoords[0], zCoords[-1]))
    xPoints = interpolateX(pointArclengths)
    zPoints = nacaz(xPoints)
    return xPoints, zPoints

# def computeColocationPoints(xCoords, zCoords, scaleX, scaleZ, chordLength):
#     midPointShift = 0.5 * (1 - scaleX) * chordLength
#     colocationXUpper = scaleX * xCoords + midPointShift
#     colocationXLower = scaleX * xCoords[-2:0:-1] + midPointShift
#     colocationZUpper = scaleZ * zCoords
#     colocationZLower = -scaleZ * zCoords[-2:0:-1]
#     return (np.concatenate((colocationXUpper, colocationXLower)),
#             np.concatenate((colocationZUpper, colocationZLower)))
            
def computeColocationPoints(xCoords, zCoords, scaleX, scaleZ, chordLength):
    midPointShift = 0.5 * (1 - scaleX) * chordLength
    colocationXUpper = scaleX * xCoords + midPointShift
    colocationXLower = scaleX * xCoords[-1:0:-1] + midPointShift
    colocationZUpper = scaleZ * zCoords
    colocationZLower = -scaleZ * zCoords[-1:0:-1]
    return (np.concatenate((colocationXUpper, colocationXLower)),
            np.concatenate((colocationZUpper, colocationZLower)))

def plotGeometry(xCoords, zCoords, colocationX, colocationZ, filename, isClosed=True):
    plt.grid(True)
    plt.axis('equal')
    plotX = np.append(xCoords, xCoords[0]) if isClosed else xCoords
    plotZ = np.append(zCoords, zCoords[0]) if isClosed else zCoords
    plt.plot(plotX, plotZ)
    plt.scatter(xCoords, zCoords, label="points", s=2)
    plt.scatter(colocationX, colocationZ, label="colocation points", s=2)
    plt.legend()
    plt.savefig(os.path.join('graphs', filename))
    plt.close()

def generateElementMatrices(numPoints, isClosed=False):
    numEffectivePoints = numPoints if not isClosed else numPoints
    numElements = (numPoints - 1) // (NUM_POINTS_IN_ELEMENT - 1) if not isClosed else ((numPoints - 1) // (NUM_POINTS_IN_ELEMENT - 1) + 1)
    connectionMatrix = np.zeros((numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
    assemblyMatrix = np.zeros((numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
    
    if not isClosed:  # NACA foil
        halfElements = numElements // 2
        for i in range(halfElements):
            start = i * 2
            connectionMatrix[i] = np.arange(start, start + NUM_POINTS_IN_ELEMENT)
            connectionMatrix[i + halfElements] = np.arange(
                start + (numPoints // 2) + 1, start + (numPoints // 2) + NUM_POINTS_IN_ELEMENT + 1)
            assemblyMatrix[i] = np.arange(start, start + NUM_POINTS_IN_ELEMENT)
            assemblyMatrix[i + halfElements] = np.arange(
                start + numPoints // 2, start + numPoints // 2 + NUM_POINTS_IN_ELEMENT)
        connectionMatrix[-1][-1] = 0
        assemblyMatrix[-1][-1] = 0
    else:  # Circle
        for i in range(numElements):
            start = i * (NUM_POINTS_IN_ELEMENT - 1)
            connectionMatrix[i] = [(start + j) % numPoints for j in range(NUM_POINTS_IN_ELEMENT)]
            assemblyMatrix[i] = [(start + j) % numPoints for j in range(NUM_POINTS_IN_ELEMENT)]
    
    return numElements, connectionMatrix, assemblyMatrix, numEffectivePoints

def computeBemInfluenceMatrices(xCoords, zCoords, colocationX, colocationZ, connectionMatrix, assemblyMatrix,
                                refCoords, refWeights, shapeFunctions, shapeDerivatives, numEffectivePoints):
    numElements = len(connectionMatrix)
    kugMatrix = np.zeros((numEffectivePoints, numEffectivePoints))
    kqgMatrix = np.zeros((numEffectivePoints, numEffectivePoints))
    gaussXCoords = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
    gaussZCoords = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
    gaussWeights = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
    gaussCosines = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
    gaussSines = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))

    for elemIdx in range(numElements):
        localKug = np.zeros((numEffectivePoints, NUM_POINTS_IN_ELEMENT))
        localKqg = np.zeros((numEffectivePoints, NUM_POINTS_IN_ELEMENT))
        elemPoints = connectionMatrix[elemIdx]
        elemAssembly = assemblyMatrix[elemIdx]
        elemAssembly = connectionMatrix[elemIdx]
        elemX = xCoords[elemPoints]
        elemZ = zCoords[elemPoints]

        for refIdx in range(NUM_REFERENCE_ELEMENT_POINTS):
            gaussX = np.sum(shapeFunctions[:, refIdx] * elemX)
            gaussZ = np.sum(shapeFunctions[:, refIdx] * elemZ)
            gaussDx = np.sum(shapeDerivatives[:, refIdx] * elemX)
            gaussDz = np.sum(shapeDerivatives[:, refIdx] * elemZ)
            deltaS = np.sqrt(gaussDx**2 + gaussDz**2)
            cosine = gaussDx / deltaS
            sine = gaussDz / deltaS

            distancesX = gaussX - colocationX[:numEffectivePoints]
            distancesZ = gaussZ - colocationZ[:numEffectivePoints]
            radii = np.sqrt(distancesX**2 + distancesZ**2)
            greensFunc = -np.log(radii) / (2 * np.pi)
            greensFuncNormal = -(sine * distancesX - cosine * distancesZ) / (2 * np.pi * radii**2)

            for ptIdx in range(NUM_POINTS_IN_ELEMENT):
                shapeVal = shapeFunctions[ptIdx, refIdx]
                weight = deltaS * refWeights[refIdx]
                localKug[:, ptIdx] += shapeVal * greensFunc * weight
                localKqg[:, ptIdx] += shapeVal * greensFuncNormal * weight

            gaussXCoords[elemIdx, refIdx] = gaussX
            gaussZCoords[elemIdx, refIdx] = gaussZ
            gaussWeights[elemIdx, refIdx] = deltaS * refWeights[refIdx]
            gaussCosines[elemIdx, refIdx] = cosine
            gaussSines[elemIdx, refIdx] = sine

        for ptIdx in range(NUM_POINTS_IN_ELEMENT):
            assemblyIdx = elemAssembly[ptIdx]
            if assemblyIdx < numEffectivePoints:
                kugMatrix[:, assemblyIdx] += localKug[:, ptIdx]
                kqgMatrix[:, assemblyIdx] += localKqg[:, ptIdx]

    return kugMatrix, kqgMatrix, gaussXCoords, gaussZCoords, gaussWeights, gaussCosines, gaussSines

def computeNormalsAndRhs(xCoords, zCoords, numPoints, U_inf, W_inf):
    normalX = np.zeros(numPoints)
    normalZ = np.zeros(numPoints)
    for i in range(numPoints):
#         print(i)
        if i == 0:
            dx = xCoords[1] - xCoords[-1]
            dz = zCoords[1] - zCoords[-1]
        elif i == numPoints - 1:
            dx = xCoords[i] - xCoords[-2]
            dz = zCoords[i] - zCoords[-2]
        elif i == (numPoints//2) and (numPoints + 1) == NUM_POINTS: #Handle trailing edge of NACA foil
            print("IN", i)
            dx = xCoords[i] - xCoords[i - 1]
            dz = zCoords[i] - zCoords[i - 1]
            print(dx, dz)
        else:
            dx = xCoords[i+1] - xCoords[i-1]
            dz = zCoords[i+1] - zCoords[i-1]
        length = np.sqrt(dx**2 + dz**2)
        normalX[i] = dz / length
        normalZ[i] = -dx / length
    print(normalX, normalZ)
    rhs = (U_inf * normalX + W_inf * normalZ)
    return normalX, normalZ, rhs
#     
# def computeNormalsAndRhs(pointXCoords, pointZCoords, numPoints, U_inf, W_inf):
#     normalX = np.zeros(numPoints)
#     normalZ = np.zeros(numPoints)
#     
#     dx = np.diff(pointXCoords)
#     dz = np.diff(pointZCoords)
#     tana = dz / dx
#     cosa_seg = 1 / np.sqrt(1 + tana**2)
#     sina_seg = tana / np.sqrt(1 + tana**2)
#     
#     for i in range(numPoints):
#         if i == 0:  # Leading edge
#             normalX[i] = 1.0  # Match sina(1) = 1.0
#             normalZ[i] = 0.0
#         elif i == numPoints // 2:  # Trailing edge (double node)
#             normalX[i] = -sina_seg[0]  # Symmetry with leading edge
#             normalZ[i] = cosa_seg[0]
#         else:
#             normalX[i] = sina_seg[min(i, len(sina_seg)-1)]  # Remove the negative sign
#             normalZ[i] = cosa_seg[min(i, len(sina_seg)-1)]
#     
#     # Lower surface (flip signs for outward normals)
#     normalX[numPoints//2 + 1:] = normalX[numPoints//2:0:-1]
#     normalZ[numPoints//2 + 1:] = -normalZ[numPoints//2:0:-1]
#     
#     rhs = -U_inf * normalX + W_inf * normalZ
#     return normalX, normalZ, rhs
    
# def computeNormalsAndRhs(xCoords, zCoords, numPoints, U_inf, W_inf):
#     normalX = np.zeros(numPoints)
#     normalZ = np.zeros(numPoints)
#     for i in range(numPoints):
# #         print(i)
#         if i == 0:
#             dx = xCoords[1] - xCoords[0]
#             dz = zCoords[1] - zCoords[0]
#         elif i == numPoints - 1:
#             dx = xCoords[0] - xCoords[-2] if numPoints == NUM_CIRCLE_POINTS - 1 else xCoords[-1] - xCoords[-2]
#             dz = zCoords[0] - zCoords[-2] if numPoints == NUM_CIRCLE_POINTS - 1 else zCoords[-1] - zCoords[-2]
#         else:
#             dx = xCoords[i+1] - xCoords[i-1]
#             dz = zCoords[i+1] - zCoords[i-1]
#         length = np.sqrt(dx**2 + dz**2)
#         normalX[i] = -dz / length
#         normalZ[i] = dx / length    # Ensure normals point inward (flip signs)
#     normalX = -normalX  # Reverse direction
#     normalZ = -normalZ  # Reverse direction
#     rhs = -U_inf * normalX - W_inf * normalZ  # Adjust RHS accordingly
#     return normalX, normalZ, rhs

def solveBem(kqgMatrix, rhs, numPoints):
    A = kqgMatrix.copy()
    gamma = np.linalg.solve(A, rhs)
    return gamma

# TODO: Reconcile forces correctly computing to validate lift
# TODO: Validate perturbationPhi and totalPhi against matlab
def computeBoundaryQuantities(kugMatrix, phiNormal, vortexGamma, vortexPhi, xCoords, zCoords, numPoints, U_inf, W_inf):
#     perturbationPhi = (kugMatrix @ phiNormal) + (-vortexGamma * vortexPhi)
#     perturbationPhi = -phiNormal + (vortexGamma * vortexPhi)
    perturbationPhi = phiNormal
#     print("phiNormal", phiNormal)
#     print("perturbationPhi", perturbationPhi)
    totalPhi = perturbationPhi + U_inf * xCoords[:numPoints] + W_inf * zCoords[:numPoints]
#     print("totalPhi", totalPhi)
    tangentX = np.zeros(numPoints)
    tangentZ = np.zeros(numPoints)
    ds = np.zeros(numPoints)
    for i in range(numPoints):
        if i == 0:
            dx = xCoords[1] - xCoords[0]
            dz = zCoords[1] - zCoords[0]
        elif i == numPoints - 1:
            dx = xCoords[0] - xCoords[-2] if numPoints == NUM_CIRCLE_POINTS - 1 else xCoords[-1] - xCoords[-2]
            dz = zCoords[0] - zCoords[-2] if numPoints == NUM_CIRCLE_POINTS - 1 else zCoords[-1] - zCoords[-2]
        else:
            dx = xCoords[i+1] - xCoords[i-1]
            dz = zCoords[i+1] - zCoords[i-1]
        length = np.sqrt(dx**2 + dz**2)
        tangentX[i] = dx / length
        tangentZ[i] = dz / length
        ds[i] = length / 2 if i != 0 and i != numPoints - 1 else length
    
#     TODO: Rewrite tangential calculation to use gauss points to find tangential velocity
#       TODO: Add vortex contribution to tangential velocity
    tangentialPertVel = np.zeros(numPoints)
    tangentialTotalVel = np.zeros(numPoints)
    for i in range(1, numPoints - 1):
        ds_i = ds[i-1] + ds[i]
        tangentialPertVel[i] = (perturbationPhi[i+1] - perturbationPhi[i-1]) / ds_i
        tangentialTotalVel[i] = (totalPhi[i+1] - totalPhi[i-1]) / ds_i
    tangentialPertVel[0] = (perturbationPhi[1] - perturbationPhi[0]) / ds[0]
    tangentialPertVel[-1] = (perturbationPhi[0] - perturbationPhi[-2]) / ds[-2]
    tangentialTotalVel[0] = (totalPhi[1] - totalPhi[0]) / ds[0]
    tangentialTotalVel[-1] = (totalPhi[0] - totalPhi[-2]) / ds[-2]
    
    
    tangentialPertVel = np.zeros(numPoints)
    tangentialTotalVel = np.zeros(numPoints)
    intrinsicCoordinates = [-1, 0, 1]
#     shapeFunctions = funf1(intrinsicCoordinates, NUM_POINTS_IN_ELEMENT)
#     shapeFunctionDerivatives = dfunf1(intrinsicCoordinates, NUM_POINTS_IN_ELEMENT)
    shapeFunctionsIntrinsic = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    shapeFunctionDerivatives = [[-1.5, -0.5, 0.5], [2, 0, -2], [-0.5, 0.5, 1.5]]
    for elementIndex in range(numElements):
        elementConnectionMatrix = connectionMatrix[elementIndex, :]
        elementPoints = elementConnectionMatrix
#         print(elementPoints)
        elementXCoordinates = xCoords[elementPoints]
        elementZCoordinates = zCoords[elementPoints]
        elementWeights = []
        if(elementIndex == numElements//2 - 1):
            elementWeights = np.array([1, 1, 0.5])
        elif(elementIndex == numElements//2):
            elementWeights = np.array([0.5, 1, 1])
        else:
            elementWeights = np.array([0.5, 1, 0.5])
        elementWeights = np.array([0.5, 1, 0.5])
        for pointIndex in range(NUM_POINTS_IN_ELEMENT):
            if(pointIndex == 0):
                shapeFunctionsIntrinsic = [1, 0, 0]
                shapeFunctionDerivatives = [-1.5, 2, -0.5]
            elif(pointIndex == 1):
                shapeFunctionsIntrinsic = [0, 1, 0]
                shapeFunctionDerivatives = [-0.5, 0,  0.5]
            else:
                shapeFunctionsIntrinsic = [0, 0, 1]
                shapeFunctionDerivatives = [0.5, -2, 1.5]
            pointXCoordinate = sum(shapeFunctionsIntrinsic[:] * elementXCoordinates[:])
            pointZCoordinate = sum(shapeFunctionsIntrinsic[:] * elementZCoordinates[:])
            pointXCoordinateDerivative = sum(shapeFunctionDerivatives[:] * elementXCoordinates[:])
            pointZCoordinateDerivative = sum(shapeFunctionDerivatives[:] * elementZCoordinates[:])
            pointDeltaArclength = np.sqrt(pointXCoordinateDerivative**2 + pointZCoordinateDerivative**2)
            pointCosine = pointXCoordinateDerivative/pointDeltaArclength
            pointSine = pointZCoordinateDerivative/pointDeltaArclength
#             print(elementConnectionMatrix[pointIndex])
            tangentialPertVelLocal = sum(shapeFunctionDerivatives * perturbationPhi[elementConnectionMatrix[:]])/pointDeltaArclength
            tangentialTotalVelLocal = tangentialPertVelLocal + U_inf*pointSine + W_inf*pointCosine
#             print("tangentialPertLocal", tangentialPertVelLocal)
#             print("tangentialTotalLocal", tangentialTotalVelLocal)
            tangentialPertVel[elementConnectionMatrix[pointIndex]] = tangentialPertVel[elementConnectionMatrix[pointIndex]] + elementWeights[pointIndex] * tangentialPertVelLocal
            tangentialTotalVel[elementConnectionMatrix[pointIndex]] = tangentialTotalVel[elementConnectionMatrix[pointIndex]] + elementWeights[pointIndex] * tangentialTotalVelLocal

    
#     print("tangentialPertVel", tangentialPertVel)
    tangentialVelX = tangentialPertVel * tangentX
    tangentialVelZ = tangentialPertVel * tangentZ
    

    perturbationPhi += vortexGamma * vortexPhi
    
    return perturbationPhi, totalPhi, tangentialVelX, tangentialVelZ, tangentialPertVel, tangentialTotalVel, ds

def computeInternalField(xCoords, zCoords, numElements, connectionMatrix, assemblyMatrix, gaussXCoords, gaussZCoords,
                         gaussWeights, gaussCosines, gaussSines, perturbationPhi, tangentialTotalVel, normalVel,
                         shapeFunctions, scaleX, scaleZ, numPointsX, numPointsZ, radius, numEffectivePoints, U_inf, W_inf):
    internalDeltaX = scaleX * radius / (numPointsX - 1)
    internalDeltaZ = scaleZ * radius / (numPointsZ - 1)
    internalXCoords = np.linspace(-scaleX * radius, scaleX * radius, numPointsX)
    internalZCoords = np.linspace(-scaleZ * radius, scaleZ * radius, numPointsZ)
    internalXGrid, internalZGrid = np.meshgrid(internalXCoords, internalZCoords, indexing='xy')

    perturbationPhiInternal = np.zeros((numPointsZ, numPointsX))
    velocityXInternal = np.zeros((numPointsZ, numPointsX))
    velocityZInternal = np.zeros((numPointsZ, numPointsX))

    for elemIdx in range(numElements):
        elemAssembly = assemblyMatrix[elemIdx]
        weights = gaussWeights[elemIdx]
        cosines = gaussCosines[elemIdx]
        sines = gaussSines[elemIdx]
        
        validAssembly = elemAssembly[elemAssembly < numEffectivePoints]
        if len(validAssembly) < NUM_POINTS_IN_ELEMENT:
            validAssembly = np.pad(validAssembly, (0, NUM_POINTS_IN_ELEMENT - len(validAssembly)), 
                                   mode='edge')
        gaussTotalVel = np.sum(shapeFunctions * tangentialTotalVel[validAssembly][:, np.newaxis], axis=0)
        
        gaussPertPhi = np.sum(shapeFunctions * perturbationPhi[validAssembly][:, np.newaxis], axis=0)
        gaussNormalVel = np.sum(shapeFunctions * normalVel[validAssembly][:, np.newaxis], axis=0)
        
        xDist = gaussXCoords[elemIdx, :, np.newaxis, np.newaxis] - internalXCoords[np.newaxis, np.newaxis, :]
        zDist = gaussZCoords[elemIdx, :, np.newaxis, np.newaxis] - internalZCoords[np.newaxis, :, np.newaxis]
        radii = np.sqrt(xDist**2 + zDist**2)
        greensFuncs = -np.log(radii) / (2 * np.pi)
        greensFuncNormals = -(sines[:, np.newaxis, np.newaxis] * xDist - cosines[:, np.newaxis, np.newaxis] * zDist) / (2 * np.pi * radii**2)
        greensDx = xDist / (2 * np.pi * radii**2)
        greensDz = zDist / (2 * np.pi * radii**2)
        greensNormalDx = (sines[:, np.newaxis, np.newaxis] / (2 * np.pi) + 2 * xDist * greensFuncNormals) / (radii**2)
        greensNormalDz = (-cosines[:, np.newaxis, np.newaxis] / (2 * np.pi) + 2 * zDist * greensFuncNormals) / (radii**2)
        
        weightsExpanded = weights[:, np.newaxis, np.newaxis]
        perturbationPhiInternal += np.sum((gaussNormalVel[:, np.newaxis, np.newaxis] * greensFuncs -
                                          gaussPertPhi[:, np.newaxis, np.newaxis] * greensFuncNormals) * weightsExpanded, axis=0)
        velocityXInternal += np.sum((gaussNormalVel[:, np.newaxis, np.newaxis] * greensDx -
                                     gaussPertPhi[:, np.newaxis, np.newaxis] * greensNormalDx) * weightsExpanded, axis=0)
        velocityZInternal += np.sum((gaussNormalVel[:, np.newaxis, np.newaxis] * greensDz -
                                     gaussPertPhi[:, np.newaxis, np.newaxis] * greensNormalDz) * weightsExpanded, axis=0)

    totalPhiInternal = perturbationPhiInternal + U_inf * internalXGrid + W_inf * internalZGrid
    velocityXTotal = velocityXInternal + U_inf
    velocityZTotal = velocityZInternal + W_inf
    return internalXGrid, internalZGrid, perturbationPhiInternal, totalPhiInternal, velocityXTotal, velocityZTotal

def computeForcesAndPressure(perturbationPhi, phiT, tangentialVelX, tangentialVelZ, tangentialTotalVel, normalX, normalZ, ds, U_inf, W_inf, numPoints):
    # Compute velocity magnitude
    velocityMag = (U_inf + tangentialVelX)**2 + (W_inf + tangentialVelZ)**2
    
    # Compute total pressure for forces (used for added mass, lift, and drag)
    U_mag_squared = U_inf**2 + W_inf**2
    C_t = 0.5 * RHO * U_mag_squared  # C(t) = p_infty/ρ + 0.5 * |U(t)|^2, assuming p_infty = 0
    total_pressure = C_t - RHO * (phiT + 0.5 * velocityMag)
    
    lift = 0
    integratedDynamicPressureForceX = 0
    dynamicPressures = []
#     TODO: Check Tangential Vel
#     tangentialVel = tangentialVelX + tangentialVelZ + (U_inf * normalX) + (W_inf * normalZ)
#     tangentialVel = np.sqrt(tangentialVelX**2 + tangentialVelZ**2) + (U_inf * normalX) + (W_inf * normalZ)
#     tangentialVel = np.sqrt(tangentialVelX**2 + tangentialVelZ**2)

#     print("tangentialTotalVel", tangentialTotalVel)
#     quit()
    for elementIndex in range(numElements):
        for gaussPointIndex in range(NUM_REFERENCE_ELEMENT_POINTS):
            tangentialVelAtGaussPoint = sum(shapeFunctions[:, gaussPointIndex] * tangentialTotalVel[connectionMatrix[elementIndex]])
            phiTAtGaussPoint = sum(shapeFunctions[:, gaussPointIndex] * phiT[connectionMatrix[elementIndex]])
            dynamicPressure = 0.5*RHO*(U_mag_squared - tangentialVelAtGaussPoint**2) + RHO*phiTAtGaussPoint
#             dynamicPressure = 0.5*RHO*(U_mag_squared - tangentialVelAtGaussPoint**2)
            dynamicPressures.append(dynamicPressure)
            lift = lift - dynamicPressure * gaussCosines[elementIndex, gaussPointIndex] * gaussWeights[elementIndex, gaussPointIndex]
            integratedDynamicPressureForceX = integratedDynamicPressureForceX - dynamicPressure * gaussSines[elementIndex, gaussPointIndex] * gaussWeights[elementIndex, gaussPointIndex]
#     lift = 0.5*RHO*(U_mag_squared - (tangentialVelX**2 + tangentialVelZ**2)) * normalZ * ds
#     print("LIFT:", lift)
#     print("integratedDynamicPressureForceX:", integratedDynamicPressureForceX)
    forceMagnitude = np.sqrt(lift**2 + integratedDynamicPressureForceX**2)
#     print("ForceMagnitude: ", forceMagnitude)
    forceX = integratedDynamicPressureForceX
    forceZ = lift
    # Compute forces using total pressure
#     forceX = 0.0
#     forceZ = 0.0
#     for i in range(numPoints):
#         forceX += total_pressure[i] * normalX[i] * ds[i]
#         forceZ -= total_pressure[i] * normalZ[i] * ds[i]
#     
#     # Compute dynamic pressure for return value (used for animations)
#     dynamic_pressure = 0.5 * RHO * velocityMag
#     dynamic_pressure = total_pressure
    
    return forceX, forceZ, dynamicPressures
    
    return forceX, forceZ, pressure
    
def computeVortexLine(xCoords, numVortexPoints):
        xCoordinateMin = min(xCoords)
        xCoordinateMax = max(xCoords)
        vortexLineLength = xCoordinateMax - xCoordinateMin
        vortexLineDeltaX = (vortexLineLength / (numVortexPoints))
        vortexLineXCoordinates = xCoordinateMin + ((0.5 * vortexLineDeltaX) + (np.arange(numVortexPoints) * vortexLineDeltaX))
        vortexLineZCoordinates = np.zeros(numVortexPoints)
        return vortexLineLength, vortexLineXCoordinates, vortexLineZCoordinates

def computeVortexVelocityNormal(pointXCoords, pointZCoords, normalX, normalZ, numPoints, vortexLineLength, 
                                vortexLineXCoordinates, vortexLineZCoordinates, numVortexPoints, integrated=True):
    # Initialize output arrays
    vortexVelocityX = np.zeros(numPoints)  # Discrete x-velocity
    vortexVelocityZ = np.zeros(numPoints)  # Discrete z-velocity
    vortexVelocityXIntegrated = np.zeros(numPoints)  # Integrated x-velocity (uvtxig equivalent)
    vortexVelocityZIntegrated = np.zeros(numPoints)  # Integrated z-velocity (vvtxig equivalent)
    vortexVelocityNormal = np.zeros(numPoints)
    vortexPhi = np.zeros(numPoints)
    vortexPhiT = np.zeros(numPoints)
    thetas = []
    # Compute velocities for each point
    for pointIndex in range(numPoints):
        theta = np.zeros(numVortexPoints)
        distanceX = pointXCoords[pointIndex] - vortexLineXCoordinates
        distanceZ = pointZCoords[pointIndex] - vortexLineZCoordinates
        distanceMagnitude = np.sqrt(distanceX**2 + distanceZ**2)
        
        # Avoid division by zero
        distanceMagnitude[distanceMagnitude < 1e-10] = 1e-10
        
        # Compute theta for all vortex points
        theta = np.arctan2(distanceZ, distanceX)
        if(max(theta) < 0):
            theta += 2*np.pi
        thetas.append(theta)
        vortexPhi[pointIndex] = ((1 / (2 * np.pi * numVortexPoints)) * np.sum(theta))
        vortexPhiT[pointIndex] = (1/(4 * np.pi**2 * np.sum(distanceMagnitude**2)))
#         print(vortexPhi[pointIndex])
#         quit()
        # Discrete velocities (non-integrated case)
        if not integrated:
            vortexVelocityX[pointIndex] = (-1 / (2 * np.pi * numVortexPoints)) * np.sum(distanceZ / (distanceMagnitude**2))
            vortexVelocityZ[pointIndex] = (1 / (2 * np.pi * numVortexPoints)) * np.sum(distanceX / (distanceMagnitude**2))
            vortexVelocityNormal[pointIndex] = (vortexVelocityX[pointIndex] * normalX[pointIndex]) + \
                                               (vortexVelocityZ[pointIndex] * normalZ[pointIndex])

        # Integrated velocities (always compute, used when integrated=True)
        vortexVelocityXIntegrated[pointIndex] = (-1 / (2 * np.pi * vortexLineLength)) * (theta[-1] - theta[0])
        
        # Z-component: Default integrated case
        if abs(pointZCoords[pointIndex]) < 1e-6:  # Leading or trailing edge (z ≈ 0)
            vortexVelocityZIntegrated[pointIndex] = (1 / (2 * np.pi * vortexLineLength)) * np.log(
                abs(pointXCoords[pointIndex] - vortexLineXCoordinates[0]) / 
                abs(pointXCoords[pointIndex] - vortexLineXCoordinates[-1])
            )
        else:
            sin_theta_start = np.sin(theta[0])
            sin_theta_end = np.sin(theta[-1])
            if abs(sin_theta_start) < 1e-10 or abs(sin_theta_end) < 1e-10:
                vortexVelocityZIntegrated[pointIndex] = 0  # Avoid log(0)
            else:
                vortexVelocityZIntegrated[pointIndex] = (1 / (2 * np.pi * vortexLineLength)) * np.log(
                    abs(sin_theta_end / sin_theta_start)
                )

        # Use integrated velocities if flag is True
        if integrated:
            vortexVelocityX[pointIndex] = vortexVelocityXIntegrated[pointIndex]
            vortexVelocityZ[pointIndex] = vortexVelocityZIntegrated[pointIndex]
            vortexVelocityNormal[pointIndex] = -((vortexVelocityX[pointIndex] * normalX[pointIndex]) + \
                                               (vortexVelocityZ[pointIndex] * normalZ[pointIndex]))

#     print("index", numPoints//2 + 1)
#     print("vortexPhi", vortexPhi)
    vortexPhi[numPoints//2 + 1] = vortexPhi[numPoints//2]
    vortexPhiT[numPoints//2 + 1] = vortexPhiT[numPoints//2]
#     quit()
#     newVortexPhi = np.concatenate((vortexPhi[0:numPoints//2], vortexPhi[numPoints//2:]))
#     vortexPhi = newVortexPhi
#     print("vortexLineXCoordinates", vortexLineXCoordinates)
#     vortexVelocityX = vortexVelocityX * 2
#     vortexVelocityZ = vortexVelocityZ * 2
#     vortexVelocityZIntegrated = vortexVelocityZIntegrated * 2
#     vortexVelocityXIntegrated = vortexVelocityXIntegrated * 2
#     vortexVelocityNormal = vortexVelocityNormal * 2
#     vortexVelocityNormal[0] = 0
    return vortexVelocityX, vortexVelocityZ, vortexVelocityNormal, vortexPhi, vortexPhiT, vortexVelocityXIntegrated, vortexVelocityZIntegrated
    
def plotForceTimeSeries(times, forcesX, forcesZ, prefix, alpha_deg):
    plt.figure(figsize=(10, 6))
    plt.plot(times, forcesZ, label='Force Z')
    plt.plot(times, forcesX, label='Force X', linestyle='--')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(f"{prefix} - Force X and Z vs Time (α = {alpha_deg}°)")
    plt.legend()
#     plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
    plt.savefig(os.path.join('graphs', f"{prefix.lower()}_force_timeseries_alpha_{alpha_deg}.png"))
    plt.close()

    
def plotForcesAlongCurrentTimeSeries(times, lift_coeffs, drag_coeffs, prefix, alpha_deg):
    plt.figure(figsize=(10, 6))
    plt.plot(times, lift_coeffs, label='Lift Coefficient')
    plt.plot(times, drag_coeffs, label='Drag Coefficient', linestyle='--')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(f"{prefix} - Lift and Drag vs Time (α = {alpha_deg}°)")
    plt.legend()
#     plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
    plt.savefig(os.path.join('graphs', f"{prefix.lower()}_lift_drag_force_timeseries_alpha_{alpha_deg}.png"))
    plt.close()

    
def plotKJLiftTimeSeries(times, kjLifts, prefix, alpha_deg):
    plt.figure(figsize=(10, 6))
    plt.plot(times, kjLifts, label='KJ Lift Force RHO * U * Gamma')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(f"{prefix} - KJ Lift Force vs Time (α = {alpha_deg}°)")
    plt.legend()
#     plt.ylim(LIFT_COEFF_MIN, LIFT_COEFF_MAX)
    plt.savefig(os.path.join('graphs', f"{prefix.lower()}_kj_lift_force_timeseries_alpha_{alpha_deg}.png"))
    plt.close()
    
def plotAddedMassTimeSeries(times, added_masses, prefix):
    plt.figure(figsize=(10, 6))
    plt.plot(times, added_masses, label='Added Mass (kg/m)')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Added Mass (kg/m)")
    plt.title(f"{prefix} - Added Mass vs Time")
    plt.legend()
    plt.ylim(ADDED_MASS_MIN, ADDED_MASS_MAX)
    plt.savefig(os.path.join('graphs', f"{prefix.lower()}_added_mass_timeseries.png"))
    plt.close()

def plotWindVelocityTimeSeries(times, u_velocities, w_velocities):
    plt.figure(figsize=(10, 6))
    plt.plot(times, u_velocities, label='X-Velocity (m/s)', color='green')
    plt.plot(times, w_velocities, label='Z-Velocity (m/s)', color='purple', linestyle='--')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Free-Stream Wind Velocity vs Time (α = 0°)")
    plt.legend()
    plt.ylim(WIND_VELOCITY_MIN, WIND_VELOCITY_MAX)
    plt.savefig(os.path.join('graphs', "wind_velocity_timeseries.png"))
    plt.close()

# Plot wind velocity once, independent of geometry
alpha_ref = np.deg2rad(0)  # Reference angle for plotting velocity
wind_u_velocities = []
wind_w_velocities = []
for t in TIME_STEPS:
    sin_omega_t = np.sin(OMEGA * t)
    u_inf = U_0 * sin_omega_t * np.cos(alpha_ref)
    w_inf = U_0 * sin_omega_t * np.sin(alpha_ref)
    wind_u_velocities.append(u_inf)
    wind_w_velocities.append(w_inf)
plotWindVelocityTimeSeries(TIME_STEPS, wind_u_velocities, wind_w_velocities)

# Main Execution for Circle (Validation)
refCoords, refWeights, shapeFunctions, shapeDerivatives = generateReferenceElementData()


if(SOLVE_CIRCLE):

    # Circle Geometry
    theta = np.linspace(0 + ((np.pi)/NUM_CIRCLE_POINTS), 2 * np.pi - ((np.pi)/NUM_CIRCLE_POINTS), NUM_CIRCLE_POINTS, endpoint=True)
    circlePointXCoords = CIRCLE_RADIUS * np.cos(theta)
    circlePointZCoords = CIRCLE_RADIUS * np.sin(theta)
    circleColocationXCoords = CIRCLE_COLOCATION_SCALING_FACTOR * circlePointXCoords
    circleColocationZCoords = CIRCLE_COLOCATION_SCALING_FACTOR * circlePointZCoords

    plotGeometry(circlePointXCoords, circlePointZCoords, circleColocationXCoords, circleColocationZCoords, 'circle_bem_nodes.png', isClosed=True)

    # BEM Setup for Circle
    numCircleElements, circleConnectionMatrix, circleAssemblyMatrix, numCirclePointsEffective = generateElementMatrices(NUM_CIRCLE_POINTS, isClosed=True)
    circleKugMatrix, circleKqgMatrix, circleGaussXCoords, circleGaussZCoords, circleGaussWeights, circleGaussCosines, circleGaussSines = computeBemInfluenceMatrices(
        circlePointXCoords, circlePointZCoords, circleColocationXCoords, circleColocationZCoords, circleConnectionMatrix, circleAssemblyMatrix,
        refCoords, refWeights, shapeFunctions, shapeDerivatives, numCirclePointsEffective)
    
    # Time Stepping for Circle (α = 0 for validation)
    alpha_deg = 0
    alpha = np.deg2rad(alpha_deg)
    circleLiftCoeffs = []
    circleDragCoeffs = []
    circleFrames = []
    circleVectorFrames = []
    circleAddedMasses = []

    # Validate at first time step (t=0)
    t_validation = TIME_STEPS[0]
    sin_omega_t = np.sin(OMEGA * t_validation)
    cos_omega_t = np.cos(OMEGA * t_validation)
    U_inf = U_0 * sin_omega_t * np.cos(alpha)
    W_inf = U_0 * sin_omega_t * np.sin(alpha)
    U_inf = U_0  * np.cos(alpha)
    W_inf = U_0  * np.sin(alpha)
    U_inf_t = U_0 * OMEGA * cos_omega_t * np.cos(alpha)
    W_inf_t = U_0 * OMEGA * cos_omega_t * np.sin(alpha)
    U_inf_t = 0
    W_inf_t = 0
    circleNormalX, circleNormalZ, circleRhs = computeNormalsAndRhs(circlePointXCoords[:numCirclePointsEffective], circlePointZCoords[:numCirclePointsEffective], numCirclePointsEffective, U_inf, W_inf)

    plt.figure()
    plt.scatter(circlePointXCoords, circlePointZCoords, c='red', label='Points')

    # Plot normal vectors (green) and vortex velocities (red)
    scale = 0.02
    for i in range(len(circlePointXCoords)):
        plt.arrow(circlePointXCoords[i], circlePointZCoords[i],
                  circleNormalX[i] * scale, circleNormalZ[i] * scale,
                  head_width=0.005, head_length=0.01, fc='green', ec='green', label='Normal Vectors' if i == 0 else "")
#                 plt.arrow(pointXCoords[i], pointZCoords[i],
#                           vortexVelocityX[i] * scale, vortexVelocityZ[i] * scale,
#                           head_width=0.005, head_length=0.01, fc='red', ec='red', label='Vortex Velocities' if i == 0 else "")

    plt.title('Circle points and normal vectors')
    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    
    circleGamma = solveBem(circleKqgMatrix, circleRhs, numCirclePointsEffective)
#     print("rhs", circleRhs)
#     print("kqg", circleKqgMatrix)
#     print(circleRhs)
#     print(circleGamma)
#     quit()
    numElements = numCircleElements
    connectionMatrix = circleConnectionMatrix
    numPointsEffective = numCirclePointsEffective
    vortexVelocityXIntegrated = np.zeros(numCirclePointsEffective)
    vortexVelocityZIntegrated = np.zeros(numCirclePointsEffective)
    normalX = circleNormalX
    normalZ = circleNormalZ
    assemblyMatrix = circleAssemblyMatrix
    gaussCosines = circleGaussCosines
    gaussSines = circleGaussSines
    gaussWeights = circleGaussWeights
    circlePerturbationPhi, _, circleTangentialVelX, circleTangentialVelZ, circleTangentialPertVelocity, circleTangentialTotalVelocity, circleDs = computeBoundaryQuantities(
        circleKugMatrix, circleGamma, 0, 0, circlePointXCoords, circlePointZCoords, numCirclePointsEffective, U_inf, W_inf)
    _, _, circleRhsPhiT = computeNormalsAndRhs(circlePointXCoords[:numCirclePointsEffective], circlePointZCoords[:numCirclePointsEffective], numCirclePointsEffective, U_inf_t, W_inf_t)
    circleGammaPhiT = solveBem(circleKqgMatrix, circleRhsPhiT, numCirclePointsEffective)
    print("tangential", circleTangentialTotalVelocity)
#     print(circleGammaPhiT)
#     print("circleConnectionMatrix", circleConnectionMatrix)
#     print("circleAssemblyMatrix", circleAssemblyMatrix)
#     print("numCirclePointsEffective", numCirclePointsEffective)
#     print("circleGammaPhiT", circleGammaPhiT)
#     print(circleKqgMatrix, circleRhsPhiT, numCirclePointsEffective)
    circlePhiT, _, _, _, _, _, _ = computeBoundaryQuantities(
        circleKugMatrix, circleGammaPhiT, 0, 0, circlePointXCoords, circlePointZCoords, numCirclePointsEffective, U_inf_t, W_inf_t)
    circlePhiT = circleKugMatrix @ circleGammaPhiT
#     print("circlePhiT", circlePhiT)
    circleForceX, circleForceZ, _ = computeForcesAndPressure(
        circlePerturbationPhi, circlePhiT, circleTangentialVelX, circleTangentialVelZ, circleTangentialTotalVelocity, circleNormalX, circleNormalZ, circleDs, U_inf, W_inf, numCirclePointsEffective)
    print("circleForceX, circleForceZ", circleForceX, circleForceZ)
    force_magnitude = np.sqrt(circleForceX**2 + circleForceZ**2)
    acceleration_magnitude = np.sqrt(U_inf_t**2 + W_inf_t**2)
    added_mass = force_magnitude / acceleration_magnitude if acceleration_magnitude != 0 else 0
    theoretical_added_mass = np.pi * RHO * CIRCLE_RADIUS**2
    print(f"Circle Added Mass (at t = 0): {added_mass:.4f} kg/m")
    print(f"Theoretical Added Mass: {theoretical_added_mass:.4f} kg/m")
    print(f"Relative Error: {abs(added_mass - theoretical_added_mass) / theoretical_added_mass * 100:.2f}%")
    quit()
    # Time series for circle with added mass computation
    for t_idx, t in tqdm(enumerate(TIME_STEPS), total=len(TIME_STEPS), desc="Circle Time Steps"):
        sin_omega_t = np.sin(OMEGA * t)
        cos_omega_t = np.cos(OMEGA * t)
        U_inf = U_0 * sin_omega_t * np.cos(alpha)
        W_inf = U_0 * sin_omega_t * np.sin(alpha)
    
    #     print(numCirclePointsEffective, len(circlePointXCoords))
    #     print("original", circlePointXCoords, circlePointZCoords)
        circleNormalX, circleNormalZ, circleRhs = computeNormalsAndRhs(circlePointXCoords[:numCirclePointsEffective], circlePointZCoords[:numCirclePointsEffective], numCirclePointsEffective, U_inf, W_inf)

        if(False):
            # Existing setup
            vortexLineLength, vortexLineXCoordinates, vortexLineZCoordinates = computeVortexLine(circleColocationXCoords, NUM_VORTEX_POINTS)
            vortexVelocityX, vortexVelocityZ, vortexVelocityNormal, vortexPhi = computeVortexVelocityNormal(
                circlePointXCoords[:numCirclePointsEffective], 
                circlePointZCoords[:numCirclePointsEffective], 
                circleNormalX, circleNormalZ, numCirclePointsEffective, 
                vortexLineLength, vortexLineXCoordinates, vortexLineZCoordinates, NUM_VORTEX_POINTS
            )

            # Compute T
            T = np.zeros(numCirclePointsEffective + 1)  # Corrected size
            T[0:numCirclePointsEffective] = np.dot(circleKugMatrix, vortexVelocityNormal)
            print("T len", len(T))

            # Double node indices
            upperPointIndex = 0
            lowerPointIndex = numCirclePointsEffective - 1

            # Continuity condition (phi[0] - phi[N-1] = 0)
            circleModifiedKqgMatrix = circleKqgMatrix.copy()
            maxKqg = np.max(np.abs(np.diag(circleModifiedKqgMatrix)))
            circleModifiedKqgMatrix[upperPointIndex, :] = 0
            circleModifiedKqgMatrix[upperPointIndex, upperPointIndex] = maxKqg
            circleModifiedKqgMatrix[upperPointIndex, lowerPointIndex] = -maxKqg  # Corrected sign
            circleRhs[upperPointIndex] = 0
            T[upperPointIndex] = 0

            # Shape function derivatives at xi = 1, -1
            shapeFunctionDerivativesOne = np.array([0.5, -2, 1.5])  # xi = 1
            shapeFunctionDerivativesMinusOne = np.array([-1.5, 2, -0.5])  # xi = -1

            # Element indices
            upperElementIndex = 0  # First element (0, 1, 2)
            lowerElementIndex = numCircleElements - 1  # Last element (N-3, N-2, N-1)

            # KJ row
            circleKJRow = np.zeros(numCirclePointsEffective)

            # Upper element (start at xi = -1)
            upperElementPointIndexes = circleConnectionMatrix[upperElementIndex]
            upperElementXCoordinates = np.sum(shapeFunctionDerivativesMinusOne * circlePointXCoords[upperElementPointIndexes])
            upperElementZCoordinates = np.sum(shapeFunctionDerivativesMinusOne * circlePointZCoords[upperElementPointIndexes])
            upperElementDeltaArclength = np.sqrt(upperElementXCoordinates**2 + upperElementZCoordinates**2)
            upperElementDeltaPhi = shapeFunctionDerivativesMinusOne / upperElementDeltaArclength
            circleKJRow[upperElementPointIndexes] = upperElementDeltaPhi

            # Lower element (end at xi = 1)
            lowerElementPointIndexes = circleConnectionMatrix[lowerElementIndex]
            lowerElementXCoordinates = np.sum(shapeFunctionDerivativesOne * circlePointXCoords[lowerElementPointIndexes])
            lowerElementZCoordinates = np.sum(shapeFunctionDerivativesOne * circlePointZCoords[lowerElementPointIndexes])
            lowerElementDeltaArclength = np.sqrt(lowerElementXCoordinates**2 + lowerElementZCoordinates**2)
            lowerElementDeltaPhi = shapeFunctionDerivativesOne / lowerElementDeltaArclength
            circleKJRow[lowerElementPointIndexes] = -lowerElementDeltaPhi

            # Tangential unit vectors
            s_upper = [upperElementXCoordinates / upperElementDeltaArclength, upperElementZCoordinates / upperElementDeltaArclength]
            s_lower = [lowerElementXCoordinates / lowerElementDeltaArclength, lowerElementZCoordinates / lowerElementDeltaArclength]

            # Compute kj_gamma using vortex velocities
            dphi_gamma_ds_upper = vortexVelocityX[upperPointIndex] * s_upper[0] + vortexVelocityZ[upperPointIndex] * s_upper[1]
            dphi_gamma_ds_lower = vortexVelocityX[lowerPointIndex] * s_lower[0] + vortexVelocityZ[lowerPointIndex] * s_lower[1]
            kj_gamma = dphi_gamma_ds_upper - dphi_gamma_ds_lower

            # KJ RHS
            kj_rhs = -(U_inf * (s_upper[0] - s_lower[0]) + W_inf * (s_upper[1] - s_lower[1]))

            # Augment the system
            A_sys = np.zeros((numCirclePointsEffective + 1, numCirclePointsEffective + 1))
            A_sys[:numCirclePointsEffective, :numCirclePointsEffective] = circleModifiedKqgMatrix
            A_sys[:numCirclePointsEffective, numCirclePointsEffective] = T[:numCirclePointsEffective]
            A_sys[numCirclePointsEffective, :numCirclePointsEffective] = circleKJRow
            A_sys[numCirclePointsEffective, numCirclePointsEffective] = kj_gamma
            rhs_sys = np.concatenate([circleRhs, [kj_rhs]])

            # Solve
            solution = np.linalg.solve(A_sys, rhs_sys)
            phi = solution[:numCirclePointsEffective]
            Gamma = solution[numCirclePointsEffective]

            # Compute lift and lift coefficient
            circleLift = RHO * U_inf * Gamma  # Removed negative sign (depends on coordinate system)
            U_infinity = np.sqrt(U_inf**2 + W_inf**2)
            chord = 2.0  # Diameter of a unit circle (radius = 1)
            C_L = (2 * Gamma) / (U_infinity * chord)
            print("Circulation Gamma:", Gamma)
            print("Lift force:", circleLift)
            print("Lift coefficient C_L:", C_L)

            # Verification
            sin_alpha = W_inf / U_infinity
            expected_C_L = 2 * np.pi * sin_alpha
            print("Expected C_L (2 * pi * sin(alpha)):", expected_C_L)
            tolerance = 0.1  # Allow 10% deviation due to numerical approximations
            if abs(C_L - expected_C_L) / expected_C_L <= tolerance:
                print("Lift verification: PASSED (within 10% of expected value)")
            else:
                print("Lift verification: FAILED (C_L deviates more than 10% from expected)")
                print("Possible issues: Check vortex line placement, angle of attack, or KJ condition implementation.")
    
        circleGamma = solveBem(circleKqgMatrix, circleRhs, numCirclePointsEffective)
        circlePerturbationPhi, circleTotalPhi, circleTangentialVelX, circleTangentialVelZ, circleTangentialTotalVel, circleDs = computeBoundaryQuantities(
            circleKugMatrix, circleGamma, circlePointXCoords, circlePointZCoords, numCirclePointsEffective, U_inf, W_inf)

    
        U_inf_t = U_0 * OMEGA * cos_omega_t * np.cos(alpha)
        W_inf_t = U_0 * OMEGA * cos_omega_t * np.sin(alpha)
        _, _, circleRhsPhiT = computeNormalsAndRhs(circlePointXCoords[:numCirclePointsEffective], circlePointZCoords[:numCirclePointsEffective], numCirclePointsEffective, U_inf_t, W_inf_t)
        circleGammaPhiT = solveBem(circleKqgMatrix, circleRhsPhiT, numCirclePointsEffective)
        circlePhiT, _, _, _, _, _ = computeBoundaryQuantities(
            circleKugMatrix, circleGammaPhiT, circlePointXCoords, circlePointZCoords, numCirclePointsEffective, U_inf_t, W_inf_t)
    
        circleForceX, circleForceZ, circlePressure = computeForcesAndPressure(
            circlePerturbationPhi, circlePhiT, circleTangentialVelX, circleTangentialVelZ, circleNormalX, circleNormalZ, circleDs, U_inf, W_inf, numCirclePointsEffective)
        force_magnitude = np.sqrt(circleForceX**2 + circleForceZ**2)
        acceleration_magnitude = np.sqrt(U_inf_t**2 + W_inf_t**2)
        added_mass = force_magnitude / acceleration_magnitude if acceleration_magnitude != 0 else 0
        circleAddedMasses.append(added_mass)
    
        flowDirX = np.cos(alpha)
        flowDirZ = np.sin(alpha)
        drag = circleForceX * flowDirX + circleForceZ * flowDirZ
        lift = -circleForceX * flowDirZ + circleForceZ * flowDirX
        area = 2 * CIRCLE_RADIUS
        dynamicPressure = 0.5 * RHO * (U_0 * sin_omega_t)**2
        circleLiftCoeff = lift / (dynamicPressure * area) if dynamicPressure != 0 else 0
        circleDragCoeff = drag / (dynamicPressure * area) if dynamicPressure != 0 else 0
        circleLiftCoeffs.append(circleLiftCoeff)
        circleDragCoeffs.append(circleDragCoeff)
    
        circleNormalVelVector = np.zeros(numCirclePointsEffective)
        circleInternalXGrid, circleInternalZGrid, circlePerturbationPhiInternal, circleTotalPhiInternal, circleVelocityXTotal, circleVelocityZTotal = computeInternalField(
            circlePointXCoords[:numCirclePointsEffective], circlePointZCoords[:numCirclePointsEffective], numCircleElements, circleConnectionMatrix, circleAssemblyMatrix,
            circleGaussXCoords, circleGaussZCoords, circleGaussWeights, circleGaussCosines, circleGaussSines, circlePerturbationPhi, circleTangentialTotalVel, circleNormalVelVector,
            shapeFunctions, SCALE_CIRCLE_INTERNAL_GRID, SCALE_CIRCLE_INTERNAL_GRID, NUM_CIRCLE_INTERNAL_POINTS, NUM_CIRCLE_INTERNAL_POINTS, 
            CIRCLE_RADIUS, numCirclePointsEffective, U_inf, W_inf)
    
        circleVelocityMagTotal = np.sqrt(circleVelocityXTotal**2 + circleVelocityZTotal**2)
        circleInvalidIndices = circleVelocityMagTotal > 1.5 * U_0
        circleVelocityXTotal[circleInvalidIndices] = np.nan
        circleVelocityZTotal[circleInvalidIndices] = np.nan
        zRange = np.linspace(np.min(circleInternalZGrid), np.max(circleInternalZGrid), 50)
        startX = np.full_like(zRange, np.min(circleInternalXGrid))
        startPoints = np.column_stack((startX, zRange))
        uMasked = np.ma.masked_where(np.isnan(circleVelocityXTotal), circleVelocityXTotal)
        vMasked = np.ma.masked_where(np.isnan(circleVelocityZTotal), circleVelocityZTotal)
    
        plt.figure(figsize=(10, 6))
        plt.streamplot(circleInternalXGrid, circleInternalZGrid, uMasked, vMasked, start_points=startPoints,
                       density=2.5, linewidth=1, color='dodgerblue', maxlength=1000, arrowsize=1)
        plt.plot(circlePointXCoords, circlePointZCoords, 'k-', label="Circle Boundary", linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.title(f'Streamlines Around Circle (t = {t:.2f}s)', fontfamily='Times New Roman', fontsize=14, pad=15)
        plt.xlabel(r'X Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
        plt.ylabel(r'Z Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        frame_filename = os.path.join('graphs', f"circle_streamlines_frame_{t_idx:03d}.png")
        plt.savefig(frame_filename, dpi=300, bbox_inches='tight')
        plt.close()
        circleFrames.append(frame_filename)
    
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(circleInternalXGrid, circleInternalZGrid, circlePerturbationPhiInternal, 
                              levels=20, cmap='viridis', vmin=PERTURBATION_PHI_MIN, vmax=PERTURBATION_PHI_MAX)
        plt.colorbar(contour, label='Perturbation Potential (m²/s)')
        skip = (slice(None, None, 5), slice(None, None, 5))
        plt.quiver(circleInternalXGrid[skip], circleInternalZGrid[skip], 
                   circleVelocityXTotal[skip], circleVelocityZTotal[skip], color='white', scale=20)
        plt.plot(circlePointXCoords, circlePointZCoords, 'k-', label="Circle Boundary", linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.title(f'Velocity Vectors and Perturbation Potential (t = {t:.2f}s)', fontfamily='Times New Roman', fontsize=14, pad=15)
        plt.xlabel(r'X Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
        plt.ylabel(r'Z Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        vector_frame_filename = os.path.join('graphs', f"circle_vectors_potential_frame_{t_idx:03d}.png")
        plt.savefig(vector_frame_filename, dpi=300, bbox_inches='tight')
        plt.close()
        circleVectorFrames.append(vector_frame_filename)

    # Plot time series for circle
    plotTimeSeries(TIME_STEPS, circleLiftCoeffs, circleDragCoeffs, 'Circle', alpha_deg)
    plotAddedMassTimeSeries(TIME_STEPS, circleAddedMasses, 'Circle')

    # Create GIF for circle streamlines
    with imageio.get_writer(os.path.join('graphs', 'circle_streamlines.gif'), mode='I', duration=0.1) as writer:
        for frame in circleFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)

    # Create GIF for circle vectors and potential
    with imageio.get_writer(os.path.join('graphs', 'circle_vectors_potential.gif'), mode='I', duration=0.1) as writer:
        for frame in circleVectorFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)

# NACA Foil Geometry (unchanged)
nacaXRaw = np.arange(0, 1 + NACA_FOIL_RESOLUTION, NACA_FOIL_RESOLUTION)
nacaZRaw = nacaz(nacaXRaw)
nPointsSurface = NUM_POINTS // 2 + 1
# print(nPointsSurface)
nacaPointXCoords, nacaPointZCoords = computeArclengthInterpolation(nacaXRaw, nacaZRaw, nPointsSurface)
# print(nacaPointXCoords, nacaPointZCoords)
nacaPointXCoords = nacaPointXCoords * NACA_CHORD_LENGTH
nacaPointZCoords = nacaPointZCoords * NACA_CHORD_LENGTH
pointXCoords = np.concatenate((nacaPointXCoords, nacaPointXCoords[-1:0:-1]))
pointZCoords = np.concatenate((nacaPointZCoords, -nacaPointZCoords[-1:0:-1]))
# print(len(pointXCoords))
colocationXCoords, colocationZCoords = computeColocationPoints(
    nacaPointXCoords, nacaPointZCoords, COLOCATION_SCALING_FACTOR_X, COLOCATION_SCALING_FACTOR_Z, NACA_CHORD_LENGTH)
    
plotGeometry(pointXCoords, pointZCoords, colocationXCoords, colocationZCoords, 'bem_nodes.png', isClosed=False)
# quit()

# BEM Setup for NACA Foil (unchanged)
numElements, connectionMatrix, assemblyMatrix, numPointsEffective = generateElementMatrices(NUM_POINTS, isClosed=False)
kugMatrix, kqgMatrix, gaussXCoords, gaussZCoords, gaussWeights, gaussCosines, gaussSines = computeBemInfluenceMatrices(
    pointXCoords, pointZCoords, colocationXCoords, colocationZCoords, connectionMatrix, assemblyMatrix,
    refCoords, refWeights, shapeFunctions, shapeDerivatives, numPointsEffective)


pointXCoordsNoDouble = np.concatenate((pointXCoords[0:nPointsSurface], pointXCoords[nPointsSurface+1:]))
pointZCoordsNoDouble = np.concatenate((pointZCoords[0:nPointsSurface], pointZCoords[nPointsSurface+1:]))
# print(pointXCoordsNoDouble, pointZCoordsNoDouble)

# Time Stepping for NACA Foil for Different Angles
for alpha_deg in ALPHA_DEGREES:
    alpha = np.deg2rad(alpha_deg)
    foilLiftCoeffs = []
    foilDragCoeffs = []
    addedMasses = []
    kjLifts = []
    forcesX = []
    forcesZ = []
    foilFrames = []
    foilVectorFrames = []
    foilTangentialTotalVelocityVectorsFrames = []
    foilTangentialTotalVelocityFrames = []
    foilTangentialPertVelocityFrames = []
    foilTotalPotentialFrames = []
    foilPertPotentialFrames = []
    
    
    for t_idx, t in tqdm(enumerate(TIME_STEPS), total=len(TIME_STEPS), desc=f"NACA Foil Time Steps (α = {alpha_deg}°)"):
        sin_omega_t = np.sin(OMEGA * t)
        cos_omega_t = np.cos(OMEGA * t)
        U_inf = U_0 * sin_omega_t * np.cos(alpha)
        W_inf = U_0 * sin_omega_t * np.sin(alpha)
#         U_inf = U_0 * np.cos(alpha)
#         W_inf = U_0 * np.sin(alpha)
        
        normalX, normalZ, rhs = computeNormalsAndRhs(pointXCoords, pointZCoords, numPointsEffective, U_inf, W_inf)

#         fullNormalX, fullNormalZ, _ = computeNormalsAndRhs(pointXCoords, pointZCoords, numPointsEffective + 1, U_inf, W_inf)
        if(True):
            # Double node indices (trailing edge at N-1 and N)
            upperPointIndex = (numPointsEffective // 2)  # N-1 (last upper surface node)
            lowerPointIndex = (numPointsEffective // 2) + 1  # N (first lower surface node)

            # Compute vortex line and velocities (updated for foil)
            vortexLineLength, vortexLineXCoordinates, vortexLineZCoordinates = computeVortexLine(colocationXCoords, NUM_VORTEX_POINTS)
#             print("Python vortexLineLength", vortexLineLength)
#             quit()

            # Compute vortex velocities
            vortexVelocityX, vortexVelocityZ, vortexVelocityNormal, vortexPhi, vortexPhiT, vortexVelocityXIntegrated, vortexVelocityZIntegrated = computeVortexVelocityNormal(
                pointXCoords, pointZCoords, normalX, normalZ, numPointsEffective,
                vortexLineLength, vortexLineXCoordinates, vortexLineZCoordinates, NUM_VORTEX_POINTS, integrated=INTEGRATE_VORTEX
            )
#             print("vortexVelocityNormal", vortexVelocityNormal)
#             plt.figure()
#             plt.scatter(pointXCoords, vortexPhi)
#             plt.show()
#             quit()
#             print("Python vortexVelocityXIntegrated", vortexVelocityXIntegrated)
#             print("Python vortexVelocityZIntegrated", vortexVelocityZIntegrated)
#             quit()

#             Plot vortex line and colocation points
#             plt.figure()
#             plt.scatter(vortexLineXCoordinates, vortexLineZCoordinates, c='blue', label='Vortex Line')
#             plt.scatter(pointXCoords, pointZCoords, c='red', label='Colocation Points')

            # Plot normal vectors (green) and vortex velocities (red)
#             scale = 0.02
#             for i in range(len(pointXCoords)):
#                 plt.arrow(pointXCoords[i], pointZCoords[i],
#                           normalX[i] * scale, normalZ[i] * scale,
#                           head_width=0.005, head_length=0.01, fc='green', ec='green', label='Normal Vectors' if i == 0 else "")
# #                 plt.arrow(pointXCoords[i], pointZCoords[i],
# #                           vortexVelocityX[i] * scale, vortexVelocityZ[i] * scale,
# #                           head_width=0.005, head_length=0.01, fc='red', ec='red', label='Vortex Velocities' if i == 0 else "")
# 
#             plt.title('Vortex Line, Colocation Points, Normal Vectors, and Vortex Velocities')
#             plt.xlabel('X Coordinate')
#             plt.ylabel('Z Coordinate')
#             plt.legend()
#             plt.grid(True)
#             plt.axis('equal')
#             plt.show()


            rhs_effective = np.dot(kugMatrix, rhs)  
            rhs_effective[lowerPointIndex] = 0
            kj_rhs = W_inf * normalX[lowerPointIndex]
            rhs_sys = np.concatenate([rhs_effective, [kj_rhs]])
#             print("Python vortexVelocityXIntegrated", vortexVelocityXIntegrated)
#             print("Python vortexVelocityZIntegrated", vortexVelocityZIntegrated)
#             print("Python vortexVelocityNormal", vortexVelocityNormal)
#             kugMatrix[lowerPointIndex, lowerPointIndex] = kugMatrix[upperPointIndex, upperPointIndex]
            # Compute T
            vortexInfluenceMatrix = np.dot(kugMatrix, vortexVelocityNormal)
#             print(np.shape(vortexInfluenceMatrix))
#             quit()
#             print("connection matrix", connectionMatrix)
#             vortexVelocityNormal = np.array([0, 1.7942, 1.1584, 0.7760, 0.4638, 0.1670, -0.1447, -0.5055, -0.9883, -1.9220, -2.2609, 2.2609, 1.9220, 0.9883, 0.5055, 0.1447, -0.1670, -0.4638, -0.7760, -1.1584, -1.7942])
#             print("Python kugMatrix", kugMatrix)
#             print("Python T", T)
            

            # Continuity condition
            modifiedKqgMatrix = kqgMatrix.copy()
            maxKqg = np.max(np.abs(np.diag(modifiedKqgMatrix)))
            modifiedKqgMatrix[lowerPointIndex, :] = 0
            modifiedKqgMatrix[lowerPointIndex, lowerPointIndex] = maxKqg
            modifiedKqgMatrix[lowerPointIndex, upperPointIndex] = -maxKqg

            # Debug: Print rhs

            # Shape function derivatives at xi = 1, -1
            shapeFunctionDerivativesOne = np.array([0.5, -2, 1.5])  # xi = 1
            shapeFunctionDerivativesMinusOne = np.array([-1.5, 2, -0.5])  # xi = -1

            # Element indices (trailing edge elements)
            upperElementIndex = (numElements // 2) - 1  # Last element of upper surface
            lowerElementIndex = (numElements // 2)  # First element of lower surface

            # Debug: Print element indices

            # KJ row
            kjRow = np.zeros(numPointsEffective)

            # Upper element (end at xi = 1)
            upperElementPointIndexes = connectionMatrix[upperElementIndex]
            upperElementXCoordinates = np.sum(shapeFunctionDerivativesOne * pointXCoords[upperElementPointIndexes])
            upperElementZCoordinates = np.sum(shapeFunctionDerivativesOne * pointZCoords[upperElementPointIndexes])
            upperElementDeltaArclength = np.sqrt(upperElementXCoordinates**2 + upperElementZCoordinates**2)
            upperElementTangentialVelocity = shapeFunctionDerivativesOne / upperElementDeltaArclength
            kjRow[upperElementPointIndexes] = upperElementTangentialVelocity

            # Lower element (start at xi = -1)
            lowerElementPointIndexes = connectionMatrix[lowerElementIndex]
            lowerElementXCoordinates = np.sum(shapeFunctionDerivativesMinusOne * pointXCoords[lowerElementPointIndexes])
            lowerElementZCoordinates = np.sum(shapeFunctionDerivativesMinusOne * pointZCoords[lowerElementPointIndexes])
            lowerElementDeltaArclength = np.sqrt(lowerElementXCoordinates**2 + lowerElementZCoordinates**2)
            lowerElementTangentialVelocity = shapeFunctionDerivativesMinusOne / lowerElementDeltaArclength
            kjRow[lowerElementPointIndexes] = lowerElementTangentialVelocity
#             print("Python kjRow", kjRow)
#             print("Python 2 * vortexVelocityZIntegrated[upperPointIndex] * normalX[upperPointIndex]", 2 * vortexVelocityZIntegrated[upperPointIndex] * normalX[upperPointIndex])
#             print("Python vortexVelocityZIntegrated", vortexVelocityZIntegrated)
#             print("Python normalX", normalX)

            # Build system matrix with KJ condition
            KSYS = np.zeros((numPointsEffective + 1, numPointsEffective + 1))
            KSYS[:numPointsEffective, :numPointsEffective] = modifiedKqgMatrix
            KSYS[:numPointsEffective, numPointsEffective] = vortexInfluenceMatrix
            KSYS[numPointsEffective, numPointsEffective] = vortexVelocityZ[upperPointIndex] * normalX[upperPointIndex]
            KSYS[numPointsEffective, :numPointsEffective] = kjRow


            # Solve
#             print("Python KSYS", KSYS)
#             print("Python rhs_sys", rhs_sys)
            solution = np.linalg.solve(KSYS, rhs_sys)
#             print("Python solution", solution)
            phiNormal = -solution[:numPointsEffective]
#             print("phiNormal", phiNormal)
            Gamma = solution[numPointsEffective]
#             print("Python phi(1:5):", phi[:5])
#             print("Python phi around TE:", phi[upperPointIndex-2:upperPointIndex+3])
#             print("Python Gamma:", Gamma)

            # Lift
            U_infinity = np.sqrt(U_inf**2 + W_inf**2)
#             lift = RHO * U_infinity * Gamma
            dynamicPressure = 0.5 * RHO * U_infinity**2
#             foilLiftCoeff = lift / (dynamicPressure * NACA_CHORD_LENGTH) if dynamicPressure != 0 else 0
            foilLiftCoeff = -2 * Gamma / (U_infinity * NACA_CHORD_LENGTH)
            liftPerUnitSpan = (RHO * Gamma * U_infinity)
            liftPerUnitSpan = foilLiftCoeff * dynamicPressure * NACA_CHORD_LENGTH
#             print("GAMMA: ", Gamma)
#             print("KUTTA LIFT COEFF", foilLiftCoeff)
#             print("KUTTA LIFT FORCE", liftPerUnitSpan)
            kjLifts.append(liftPerUnitSpan)
#             print("phiNormal", phiNormal)
#             quit()
#             print("Lift Per Unit Span", liftPerUnitSpan)
#            *****
#             Lift acts perpendicular to the direction of the flow
#             Since in this simulation, the angle of attack is fix

#             print("X, Z components of lift", liftPerUnitSpan * -np.sin(alpha), liftPerUnitSpan * np.cos(alpha))
#             print("Python U_infinity:", U_infinity)
#             print("Python lift:", lift)
#             print("Python dynamicPressure:", dynamicPressure)
#             print("Angle of attack: ", alpha_deg)
#             print("Foil Lift Coefficient:", foilLiftCoeff)    
#             print("Small Angle Approximation Lift Coefficient:", 2 * np.pi * alpha)      
                        # Compute drag (unchanged)
#             print("vortexPhi", vortexPhi)
            perturbationPhi, totalPhi, tangentialVelX, tangentialVelZ, tangentialPertVel, tangentialTotalVel, ds = computeBoundaryQuantities(
                kugMatrix, phiNormal, Gamma, vortexPhi, pointXCoords, pointZCoords, numPointsEffective, U_inf, W_inf)
#             print("tangentialPertVel", tangentialPertVel)
#             quit()
#             print("GHAMMA", Gamma)
#             print("vortexVelocityXIntegrated", vortexVelocityXIntegrated)
#             print("vortexVelocityZIntegrated", vortexVelocityZIntegrated)
#             print("Tangential Pert Vel", tangentialPertVel)
#             quit()
#             tangentialVelX += Gamma * (vortexVelocityXIntegrated * normalX) + Gamma * (vortexVelocityZIntegrated * normalZ)
#             vortexTangentialVel = abs(Gamma * ((vortexVelocityXIntegrated * normalZ) + (vortexVelocityZIntegrated * normalX)))
#             print("perturbationPhi", perturbationPhi)
#             quit()
#             print("tangentialVel", np.sqrt(tangentialVelX**2 + tangentialVelZ**2))
#             tangentialTotalVel += np.sqrt(tangentialVelX**2 + tangentialVelZ**2)
#             print("tangentialTotalVel", tangentialTotalVel)

# PLot tangential velocity given normal vector. Find vector perpendicular to normal.
# To find a perpendicular vector to a given vector, you can use the formula by transposing the components and changing the sign of one. For example, if you have a vector v=(a,b), a perpendicular vector can be v 
# ⊥
# ​	
#  =(−b,a) in two dimensions.


#             quit()

            plt.figure()
            plt.plot(pointXCoords[:numPointsEffective//2+1], tangentialTotalVel[:numPointsEffective//2+1], label="upper")
            plt.plot(pointXCoords[numPointsEffective//2+1:], tangentialTotalVel[numPointsEffective//2+1:], label="lower")
            plt.legend()
            plt.xlabel("equal")
            plt.ylabel("Tangential Velocity m/s")
            plt.title("Tangential Total Velocity Along Boundary")
            foilTangentialTotalVelocityFrameFilename = os.path.join('graphs', f"foil_tangential_total_velocity_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(foilTangentialTotalVelocityFrameFilename)
            foilTangentialTotalVelocityFrames.append(foilTangentialTotalVelocityFrameFilename)
            plt.close()
            
            plt.figure()
            plt.plot(pointXCoords[:numPointsEffective//2+1], tangentialPertVel[:numPointsEffective//2+1], label="upper")
            plt.plot(pointXCoords[numPointsEffective//2+1:], tangentialPertVel[numPointsEffective//2+1:], label="lower")
            plt.legend()
            plt.xlabel("equal")
            plt.ylabel("Tangential Perturbation Velocity m/s")
            plt.title("Tangential Perturbation Velocity Along Boundary")
            foilTangentialPertVelocityFrameFilename = os.path.join('graphs', f"foil_tangential_perturbation_velocity_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(foilTangentialPertVelocityFrameFilename)
            foilTangentialPertVelocityFrames.append(foilTangentialPertVelocityFrameFilename)
            plt.close()
            
            plt.figure()
            plt.plot(pointXCoords[:numPointsEffective//2+1], totalPhi[:numPointsEffective//2+1], label="upper")
            plt.plot(pointXCoords[numPointsEffective//2+1:], totalPhi[numPointsEffective//2+1:], label="lower")
            plt.legend()
            plt.xlabel("equal")
            plt.ylabel("Total Potential")
            plt.title("Total Potential Along Boundary")
            foilTotalPotentialFrameFilename = os.path.join('graphs', f"foil_total_potential_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(foilTotalPotentialFrameFilename)
            foilTotalPotentialFrames.append(foilTotalPotentialFrameFilename)
            plt.close()
            
            plt.figure()
            plt.plot(pointXCoords[:numPointsEffective//2+1], perturbationPhi[:numPointsEffective//2+1], label="upper")
            plt.plot(pointXCoords[numPointsEffective//2+1:], perturbationPhi[numPointsEffective//2+1:], label="lower")
            plt.legend()
            plt.xlabel("equal")
            plt.ylabel("Perturbation Potential")
            plt.title("Perturbation Potential Along Boundary")
            foilPertPotentialFrameFilename = os.path.join('graphs', f"foil_perturbation_potential_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(foilPertPotentialFrameFilename)
            foilPertPotentialFrames.append(foilPertPotentialFrameFilename)
            plt.close()

            U_inf_t = U_0 * OMEGA * cos_omega_t * np.cos(alpha)
            W_inf_t = U_0 * OMEGA * cos_omega_t * np.sin(alpha)
#             U_inf_t = 0
#             W_inf_t = 0
            _, _, rhsPhiNormalT = computeNormalsAndRhs(pointXCoords, pointZCoords, numPointsEffective, U_inf_t, W_inf_t)
            rhsPhiNormalT[lowerPointIndex] = 0
            rhsPhiNormalTKJ = W_inf_t * normalX[lowerPointIndex]
            rhsPhiNormalTSys = np.concatenate([rhsPhiNormalT, [rhsPhiNormalTKJ]])
            phiNormalTSolution = np.linalg.solve(KSYS, rhsPhiNormalTSys)
#             print("Python solution", solution)
            phiNormalT = -phiNormalTSolution[:numPointsEffective]
            GammaT = phiNormalTSolution[numPointsEffective]
#             print("GammaT", GammaT)
#             quit()
            phiT, _, _, _, _, _, _ = computeBoundaryQuantities(
                kugMatrix, phiNormalT, 0, vortexPhiT, pointXCoords, pointZCoords, numPointsEffective, U_inf_t, W_inf_t)
            phiT = kugMatrix @ phiT
            scaledVortexPhiT = Gamma**2 * vortexPhiT
            phiT += scaledVortexPhiT
#             print(scaledVortexPhiT)
            forceX, forceZ, pressure = computeForcesAndPressure(
                perturbationPhi, phiT, tangentialVelX, tangentialVelZ, tangentialTotalVel, normalX, normalZ, ds, U_inf, W_inf, numPointsEffective)
            flowDirX = np.cos(alpha)
            flowDirZ = np.sin(alpha)
            currentUnitVector = [flowDirX, flowDirZ]
            currentPerpendicularUnitVector = [-flowDirZ, flowDirX]
#             print("forceX", forceX)
#             print("forceZ", forceZ)
            forcesX.append(forceX)
            forcesZ.append(forceZ)
#             quit()
            foilDragCoeff = forceX * currentUnitVector[0] + forceZ * currentUnitVector[1]
            foilLiftCoeff = forceX * currentPerpendicularUnitVector[0] + forceZ * currentPerpendicularUnitVector[1]
#             print("foilDragCoeff (Force parallel to current)", foilDragCoeff)
#             print("foilLiftCoeff (Force perpendicular to current)", foilLiftCoeff)
#             forceDragCoeff = forceX
#             forceLiftCoeff = forceZ
#            Forces due to unsteady flow (added mass)
#           Including the KJ circulation
#             flowDirX = np.cos(alpha)
#             flowDirZ = np.sin(alpha)
#             drag = forceX * flowDirX + forceZ * flowDirZ
#             foilDragCoeff = drag / (dynamicPressure * NACA_CHORD_LENGTH) if dynamicPressure != 0 else 0
# 
#             drag = forceX * flowDirX
# #             foilDragCoeff = drag / (dynamicPressure * NACA_CHORD_LENGTH)
#             lift = forceX
#             drag = lift
#             lift = 2 * Gamma
#             foilLiftCoeff = lift
#             foilDrafCoeff = drag
#             foilLiftCoeff = lift / (dynamicPressure * NACA_CHORD_LENGTH)
            
            forceMagnitude = np.sqrt(forceX**2 + forceZ**2)
            dragLiftMagnitude = np.sqrt(foilDragCoeff**2 + foilLiftCoeff**2)
#             print("FORCE vs DRAG CHECK", forceMagnitude, dragLiftMagnitude)
            accelerationMagnitude = np.sqrt(U_inf_t**2 + W_inf_t**2)
            addedMasses.append(forceMagnitude / accelerationMagnitude if accelerationMagnitude != 0 else 0)
            
            # Store coefficients
            foilLiftCoeffs.append(foilLiftCoeff)
            foilDragCoeffs.append(foilDragCoeff)
            
            plt.figure()
            plt.scatter(vortexLineXCoordinates, vortexLineZCoordinates, c='blue', label='Vortex Line')
            plt.scatter(pointXCoords, pointZCoords, c='red', label='Geometry Points')
# 
# #             Plot normal vectors (green) and vortex velocities (red)
            scale = 0.02
            for i in range(len(pointXCoords)):
                plt.arrow(pointXCoords[i], pointZCoords[i],
                          tangentialTotalVel[i] * -normalZ[i] * scale, tangentialTotalVel[i] * normalX[i] * scale,
                           fc='green', ec='green', label='Tangential Velocities' if i == 0 else "")
            plt.arrow(NACA_CHORD_LENGTH/2, 0,
                forceX * scale * 0.01, forceZ * scale * 0.01,
                   fc='pink', ec='pink', label='Force Vector')
#                 plt.arrow(pointXCoords[i], pointZCoords[i],
#                           vortexVelocityX[i] * scale, vortexVelocityZ[i] * scale,
#                           head_width=0.005, head_length=0.01, fc='red', ec='red', label='Vortex Velocities' if i == 0 else "")

            plt.title('Vortex Line, Geometry Points, and Tangential Total Velocities')
            plt.xlabel('X Coordinate')
            plt.ylabel('Z Coordinate')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            foilTangentialTotalVelocityVectorsFrameFilename = os.path.join('graphs', f"foil_tangential_total_velocity_vectors_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(foilTangentialTotalVelocityVectorsFrameFilename)
            foilTangentialTotalVelocityVectorsFrames.append(foilTangentialTotalVelocityVectorsFrameFilename)
            plt.close()
#             plt.show()

            # Internal field (unchanged)
#             normalVelVector = np.zeros(numPointsEffective)
#             internalXGrid, internalZGrid, perturbationPhiInternal, totalPhiInternal, velocityXTotal, velocityZTotal = computeInternalField(
#                 pointXCoords, pointZCoords, numElements, connectionMatrix, assemblyMatrix, gaussXCoords, gaussZCoords,
#                 gaussWeights, gaussCosines, gaussSines, perturbationPhi, tangentialTotalVel, normalVelVector, shapeFunctions,
#                 SCALE_NACA_INTERNAL_GRID_X, SCALE_NACA_INTERNAL_GRID_Z, NUM_NACA_INTERNAL_POINTS_X, NUM_NACA_INTERNAL_POINTS_Z, 
#                 NACA_CHORD_LENGTH, numPointsEffective, U_inf, W_inf)
#         
#             velocityMagTotal = np.sqrt(velocityXTotal**2 + velocityZTotal**2)
#             invalidIndices = velocityMagTotal > 1.5 * U_0
#             velocityXTotal[invalidIndices] = np.nan
#             velocityZTotal[invalidIndices] = np.nan
#             zRange = np.linspace(np.min(internalZGrid), np.max(internalZGrid), 50)
#             startX = np.full_like(zRange, np.min(internalXGrid))
#             startPoints = np.column_stack((startX, zRange))
#             uMasked = np.ma.masked_where(np.isnan(velocityXTotal), velocityXTotal)
#             vMasked = np.ma.masked_where(np.isnan(velocityZTotal), velocityZTotal)
#             
        # Compute drag (unchanged)
#         perturbationPhi, totalPhi, tangentialVelX, tangentialVelZ, tangentialTotalVel, ds = computeBoundaryQuantities(
#             kugMatrix, phi, Gamma, pointXCoords, pointZCoords, numPointsEffective, U_inf, W_inf)
#         
#         U_inf_t = U_0 * OMEGA * cos_omega_t * np.cos(alpha)
#         W_inf_t = U_0 * OMEGA * cos_omega_t * np.sin(alpha)
#         _, _, rhsPhiT = computeNormalsAndRhs(pointXCoords, pointZCoords, numPointsEffective, U_inf_t, W_inf_t)
#         gammaPhiT = solveBem(kqgMatrix, rhsPhiT, numPointsEffective)
#         phiT, _, _, _, _, _ = computeBoundaryQuantities(
#             kugMatrix, gammaPhiT, Gamma, pointXCoords, pointZCoords, numPointsEffective, U_inf_t, W_inf_t)
#         
#         forceX, forceZ, pressure = computeForcesAndPressure(
#             perturbationPhi, phiT, tangentialVelX, tangentialVelZ, normalX, normalZ, ds, U_inf, W_inf, numPointsEffective)
#         
#         flowDirX = np.cos(alpha)
#         flowDirZ = np.sin(alpha)
#         drag = forceX * flowDirX + forceZ * flowDirZ
#         foilDragCoeff = drag / (dynamicPressure * NACA_CHORD_LENGTH) if dynamicPressure != 0 else 0
# 
#         # Store coefficients
#         foilLiftCoeffs.append(foilLiftCoeff)
#         foilDragCoeffs.append(foilDragCoeff)
# 
#         # Internal field (unchanged)
#         normalVelVector = np.zeros(numPointsEffective)
#         internalXGrid, internalZGrid, perturbationPhiInternal, totalPhiInternal, velocityXTotal, velocityZTotal = computeInternalField(
#             pointXCoords, pointZCoords, numElements, connectionMatrix, assemblyMatrix, gaussXCoords, gaussZCoords,
#             gaussWeights, gaussCosines, gaussSines, perturbationPhi, tangentialTotalVel, normalVelVector, shapeFunctions,
#             SCALE_NACA_INTERNAL_GRID_X, SCALE_NACA_INTERNAL_GRID_Z, NUM_NACA_INTERNAL_POINTS_X, NUM_NACA_INTERNAL_POINTS_Z, 
#             NACA_CHORD_LENGTH, numPointsEffective, U_inf, W_inf)
#         
#         velocityMagTotal = np.sqrt(velocityXTotal**2 + velocityZTotal**2)
#         invalidIndices = velocityMagTotal > 1.5 * U_0
#         velocityXTotal[invalidIndices] = np.nan
#         velocityZTotal[invalidIndices] = np.nan
#         zRange = np.linspace(np.min(internalZGrid), np.max(internalZGrid), 50)
#         startX = np.full_like(zRange, np.min(internalXGrid))
#         startPoints = np.column_stack((startX, zRange))
#         uMasked = np.ma.masked_where(np.isnan(velocityXTotal), velocityXTotal)
#         vMasked = np.ma.masked_where(np.isnan(velocityZTotal), velocityZTotal)

        if(GRAPH_MAPS):
            plt.figure(figsize=(10, 6))
            plt.streamplot(internalXGrid, internalZGrid, uMasked, vMasked, start_points=startPoints,
                           density=2.5, linewidth=1, color='dodgerblue', maxlength=1000, arrowsize=1)
            plt.plot(pointXCoords[:numPointsEffective], pointZCoords[:numPointsEffective], 'k-', label="Foil Upper", linewidth=2)
            plt.plot(pointXCoords[:numPointsEffective//2+1], -pointZCoords[:numPointsEffective//2+1], 'k-', label="Foil Lower", linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axis('equal')
            plt.title(f'Streamlines Around Foil (α = {alpha_deg}°, t = {t:.2f}s)', fontfamily='Times New Roman', fontsize=14, pad=15)
            plt.xlabel(r'X Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
            plt.ylabel(r'Z Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
            plt.legend(loc='upper right', fontsize=10)
            frame_filename = os.path.join('graphs', f"foil_streamlines_alpha_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(frame_filename, dpi=300, bbox_inches='tight')
            plt.close()
            foilFrames.append(frame_filename)
        
            plt.figure(figsize=(10, 6))
            contour = plt.contourf(internalXGrid, internalZGrid, perturbationPhiInternal, 
                                  levels=20, cmap='viridis', vmin=PERTURBATION_PHI_MIN, vmax=PERTURBATION_PHI_MAX)
            plt.colorbar(contour, label='Perturbation Potential (m²/s)')
            skip = (slice(None, None, 5), slice(None, None, 5))
            plt.quiver(internalXGrid[skip], internalZGrid[skip], 
                       velocityXTotal[skip], velocityZTotal[skip], color='white', scale=20)
            plt.plot(pointXCoords[:numPointsEffective], pointZCoords[:numPointsEffective], 'k-', label="Foil Upper", linewidth=2)
            plt.plot(pointXCoords[:numPointsEffective//2+1], -pointZCoords[:numPointsEffective//2+1], 'k-', label="Foil Lower", linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axis('equal')
            plt.title(f'Velocity Vectors and Perturbation Potential (α = {alpha_deg}°, t = {t:.2f}s)', fontfamily='Times New Roman', fontsize=14, pad=15)
            plt.xlabel(r'X Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
            plt.ylabel(r'Z Coordinate (m)', fontfamily='Times New Roman', fontsize=12)
            plt.legend(loc='upper right', fontsize=10)
            vector_frame_filename = os.path.join('graphs', f"foil_vectors_potential_alpha_{alpha_deg}_frame_{t_idx:03d}.png")
            plt.savefig(vector_frame_filename, dpi=300, bbox_inches='tight')
            plt.close()
            foilVectorFrames.append(vector_frame_filename)
    
#     print("max forceLiftCoeff", np.max(foilLiftCoeffs))
#     print("max forceLiftCoeff", np.max(foilDragCoeffs))
    plotForcesAlongCurrentTimeSeries(TIME_STEPS, foilLiftCoeffs, foilDragCoeffs, f"Foil_alpha_{alpha_deg}", alpha_deg)
    plotForceTimeSeries(TIME_STEPS, forcesX, forcesZ, f"Foil_alpha_{alpha_deg}", alpha_deg)
    plotKJLiftTimeSeries(TIME_STEPS, kjLifts, f"Foil_alpha_{alpha_deg}", alpha_deg)
    plotAddedMassTimeSeries(TIME_STEPS, addedMasses, f"Foil_alpha_{alpha_deg}")
    
    with imageio.get_writer(os.path.join('graphs', f"foil_tangential_total_velocity_vectors_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
        for frame in foilTangentialTotalVelocityVectorsFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)
    with imageio.get_writer(os.path.join('graphs', f"foil_tangential_total_velocity_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
        for frame in foilTangentialTotalVelocityFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)
    with imageio.get_writer(os.path.join('graphs', f"foil_tangential_perturbation_velocity_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
        for frame in foilTangentialPertVelocityFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)
    with imageio.get_writer(os.path.join('graphs', f"foil_total_potential_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
        for frame in foilTotalPotentialFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)
    with imageio.get_writer(os.path.join('graphs', f"foil_perturbation_potential_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
        for frame in foilPertPotentialFrames:
            image = imageio.imread(frame)
            writer.append_data(image)
            os.remove(frame)
    if(GRAPH_MAPS):
        with imageio.get_writer(os.path.join('graphs', f"foil_streamlines_alpha_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
            for frame in foilFrames:
                image = imageio.imread(frame)
                writer.append_data(image)
                os.remove(frame)
    
        with imageio.get_writer(os.path.join('graphs', f"foil_vectors_potential_alpha_{alpha_deg}.gif"), mode='I', duration=0.1) as writer:
            for frame in foilVectorFrames:
                image = imageio.imread(frame)
                writer.append_data(image)
                os.remove(frame)