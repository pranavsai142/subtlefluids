from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants (unchanged)
RHO = 1000
NUM_REFERENCE_ELEMENT_POINTS = 80
NUM_POINTS_IN_ELEMENT = 3
NACA_FOIL_RESOLUTION = 0.001
NACA_0012_FOIL_TYPE = "0012"
COLOCATION_SCALING_FACTOR_X = 0.995
COLOCATION_SCALING_FACTOR_Z = 0.95

class Geometry:
    def __init__(self, pointXCoords, pointZCoords, colocationXCoords, colocationZCoords, hasTrailingEdge=True):
        self.pointXCoords = np.array(pointXCoords)
        self.pointZCoords = np.array(pointZCoords)
        self.colocationXCoords = np.array(colocationXCoords)
        self.colocationZCoords = np.array(colocationZCoords)
        self.numPoints = len(self.pointXCoords)
        self.hasTrailingEdge = hasTrailingEdge
        self.generateNormals()
        self.generateReferenceElementData()

    # Keep the rest of the methods exactly as provided in the original code
    def generateNormals(self):
        self.normalX = np.zeros(self.numPoints)
        self.normalZ = np.zeros(self.numPoints)
        for i in range(self.numPoints):
            if i == 0:
                dx = self.pointXCoords[1] - self.pointXCoords[-1]
                dz = self.pointZCoords[1] - self.pointZCoords[-1]
            elif i == self.numPoints - 1:
                dx = self.pointXCoords[i] - self.pointXCoords[-2]
                dz = self.pointZCoords[i] - self.pointZCoords[-2]
            else:
                dx = self.pointXCoords[i+1] - self.pointXCoords[i-1]
                dz = self.pointZCoords[i+1] - self.pointZCoords[i-1]
            length = np.sqrt(dx**2 + dz**2)
            self.normalX[i] = dz / length
            self.normalZ[i] = -dx / length

    def generateReferenceElementData(self):
        self.referenceCoords, self.referenceWeights = np.polynomial.legendre.leggauss(NUM_REFERENCE_ELEMENT_POINTS)
        self.generateShapeFunctions()
        self.generateShapeDerivatives()

    def generateShapeFunctions(self):
        self.shapeFunctions = np.zeros((NUM_POINTS_IN_ELEMENT, len(self.referenceCoords)))
        if NUM_POINTS_IN_ELEMENT == 2:
            self.shapeFunctions[0, :] = 0.5*(1-self.referenceCoords)
            self.shapeFunctions[1, :] = 0.5*(1+self.referenceCoords)
        elif NUM_POINTS_IN_ELEMENT == 3:
            self.shapeFunctions[0, :] = 0.5*self.referenceCoords*(self.referenceCoords-1)
            self.shapeFunctions[1, :] = 1-self.referenceCoords*self.referenceCoords
            self.shapeFunctions[2, :] = 0.5*self.referenceCoords*(self.referenceCoords+1)

    def generateShapeDerivatives(self):
        self.shapeDerivatives = np.zeros((NUM_POINTS_IN_ELEMENT, len(self.referenceCoords)))
        if NUM_POINTS_IN_ELEMENT == 2:
            self.shapeDerivatives[0, :] = -0.5
            self.shapeDerivatives[1, :] = 0.5
        elif NUM_POINTS_IN_ELEMENT == 3:
            self.shapeDerivatives[0, :] = self.referenceCoords-0.5
            self.shapeDerivatives[1, :] = -2*self.referenceCoords
            self.shapeDerivatives[2, :] = self.referenceCoords+0.5

    def setElementMatrices(self, connectionMatrix, assemblyMatrix):
        self.connectionMatrix = connectionMatrix
        self.assemblyMatrix = assemblyMatrix
        self.generateInfluenceMatrices()
        if self.hasTrailingEdge:
            self.generateVortex()

    # Include all other methods exactly as in the original code (generateVortex, generateVortexVelocity, etc.)
    def generateVortex(self):
        self.numVortexPoints = len(self.pointXCoords) // 2
        xCoordinateMin = min(self.colocationXCoords)
        xCoordinateMax = max(self.colocationXCoords)
        self.vortexLineLength = xCoordinateMax - xCoordinateMin
        vortexLineDeltaX = (self.vortexLineLength / (self.numVortexPoints))
        self.vortexLineXCoordinates = xCoordinateMin + ((0.5 * vortexLineDeltaX) + (np.arange(self.numVortexPoints) * vortexLineDeltaX))
        self.vortexLineZCoordinates = np.zeros(self.numVortexPoints)
        self.generateVortexVelocity()
        self.generateModifiedKqgMatrix()

    def generateVortexVelocity(self):
        integrated = True
        vortexVelocityX = np.zeros(self.numPoints)
        vortexVelocityZ = np.zeros(self.numPoints)
        vortexVelocityXIntegrated = np.zeros(self.numPoints)
        vortexVelocityZIntegrated = np.zeros(self.numPoints)
        vortexVelocityNormal = np.zeros(self.numPoints)
        vortexPhi = np.zeros(self.numPoints)
        vortexPhiT = np.zeros(self.numPoints)
        for pointIndex in range(self.numPoints):
            theta = np.zeros(self.numVortexPoints)
            distanceX = self.pointXCoords[pointIndex] - self.vortexLineXCoordinates
            distanceZ = self.pointZCoords[pointIndex] - self.vortexLineZCoordinates
            distanceMagnitude = np.sqrt(distanceX**2 + distanceZ**2)
            distanceMagnitude[distanceMagnitude < 1e-10] = 1e-10
            theta = np.arctan2(distanceZ, distanceX)
            if max(theta) < 0:
                theta += 2*np.pi
            vortexPhi[pointIndex] = ((1 / (2 * np.pi * self.numVortexPoints)) * np.sum(theta))
            vortexPhiT[pointIndex] = (1/(4 * np.pi**2 * np.sum(distanceMagnitude**2)))
            if not integrated:
                vortexVelocityX[pointIndex] = (-1 / (2 * np.pi * self.numVortexPoints)) * np.sum(distanceZ / (distanceMagnitude**2))
                vortexVelocityZ[pointIndex] = (1 / (2 * np.pi * self.numVortexPoints)) * np.sum(distanceX / (distanceMagnitude**2))
                vortexVelocityNormal[pointIndex] = (vortexVelocityX[pointIndex] * self.normalX[pointIndex]) + \
                                                   (vortexVelocityZ[pointIndex] * self.normalZ[pointIndex])
            vortexVelocityXIntegrated[pointIndex] = (-1 / (2 * np.pi * self.vortexLineLength)) * (theta[-1] - theta[0])
            if abs(self.pointZCoords[pointIndex]) < 1e-6:
                vortexVelocityZIntegrated[pointIndex] = (1 / (2 * np.pi * self.vortexLineLength)) * np.log(
                    abs(self.pointXCoords[pointIndex] - self.vortexLineXCoordinates[0]) / 
                    abs(self.pointXCoords[pointIndex] - self.vortexLineXCoordinates[-1])
                )
            else:
                sin_theta_start = np.sin(theta[0])
                sin_theta_end = np.sin(theta[-1])
                if abs(sin_theta_start) < 1e-10 or abs(sin_theta_end) < 1e-10:
                    vortexVelocityZIntegrated[pointIndex] = 0
                else:
                    vortexVelocityZIntegrated[pointIndex] = (1 / (2 * np.pi * self.vortexLineLength)) * np.log(
                        abs(sin_theta_end / sin_theta_start)
                    )
            if integrated:
                vortexVelocityX[pointIndex] = vortexVelocityXIntegrated[pointIndex]
                vortexVelocityZ[pointIndex] = vortexVelocityZIntegrated[pointIndex]
                vortexVelocityNormal[pointIndex] = -((vortexVelocityX[pointIndex] * self.normalX[pointIndex]) + \
                                                   (vortexVelocityZ[pointIndex] * self.normalZ[pointIndex]))
        vortexPhi[self.numPoints//2 + 1] = vortexPhi[self.numPoints//2]
        vortexPhiT[self.numPoints//2 + 1] = vortexPhiT[self.numPoints//2]
        self.vortexPhi = vortexPhi
        self.vortexPhiT = vortexPhiT
        self.vortexVelocityX = vortexVelocityX
        self.vortexVelocityZ = vortexVelocityZ
        self.vortexVelocityNormal = vortexVelocityNormal

    def generateModifiedKqgMatrix(self):
        upperPointIndex = (self.numPoints // 2)
        lowerPointIndex = (self.numPoints // 2) + 1
        vortexInfluenceMatrix = np.dot(self.kugMatrix, self.vortexVelocityNormal)
        vortexInfluenceMatrix[upperPointIndex] = 0
        vortexInfluenceMatrix[lowerPointIndex] = 0
        modifiedKqgMatrix = self.kqgMatrix.copy()
        maxKqg = np.max(np.abs(np.diag(modifiedKqgMatrix)))
        modifiedKqgMatrix[lowerPointIndex, :] = 0
        modifiedKqgMatrix[lowerPointIndex, lowerPointIndex] = maxKqg
        modifiedKqgMatrix[lowerPointIndex, upperPointIndex] = -maxKqg
        shapeFunctionDerivativesOne = np.array([0.5, -2, 1.5])
        shapeFunctionDerivativesMinusOne = np.array([-1.5, 2, -0.5])
        numElements = len(self.connectionMatrix)
        upperElementIndex = (numElements // 2) - 1
        lowerElementIndex = (numElements // 2)
        kjRow = np.zeros(self.numPoints)
        upperElementPointIndexes = self.connectionMatrix[upperElementIndex]
        upperElementXCoordinates = np.sum(shapeFunctionDerivativesOne * self.pointXCoords[upperElementPointIndexes])
        upperElementZCoordinates = np.sum(shapeFunctionDerivativesOne * self.pointZCoords[upperElementPointIndexes])
        upperElementDeltaArclength = np.sqrt(upperElementXCoordinates**2 + upperElementZCoordinates**2)
        upperElementTangentialVelocity = shapeFunctionDerivativesOne / upperElementDeltaArclength
        kjRow[upperElementPointIndexes] = upperElementTangentialVelocity
        lowerElementPointIndexes = self.connectionMatrix[lowerElementIndex]
        lowerElementXCoordinates = np.sum(shapeFunctionDerivativesMinusOne * self.pointXCoords[lowerElementPointIndexes])
        lowerElementZCoordinates = np.sum(shapeFunctionDerivativesMinusOne * self.pointZCoords[lowerElementPointIndexes])
        lowerElementDeltaArclength = np.sqrt(lowerElementXCoordinates**2 + lowerElementZCoordinates**2)
        lowerElementTangentialVelocity = shapeFunctionDerivativesMinusOne / lowerElementDeltaArclength
        kjRow[lowerElementPointIndexes] = lowerElementTangentialVelocity
        KSYS = np.zeros((self.numPoints + 1, self.numPoints + 1))
        KSYS[:self.numPoints, :self.numPoints] = modifiedKqgMatrix
        KSYS[:self.numPoints, self.numPoints] = vortexInfluenceMatrix
        KSYS[self.numPoints, self.numPoints] = 2 * self.vortexVelocityZ[upperPointIndex] * self.normalX[upperPointIndex]
        KSYS[self.numPoints, :self.numPoints] = kjRow
        self.modifiedKqgMatrix = KSYS

    def generateInfluenceMatrices(self):
        numElements = len(self.connectionMatrix)
        self.kugMatrix = np.zeros((self.numPoints, self.numPoints))
        self.kqgMatrix = np.zeros((self.numPoints, self.numPoints))
        self.gaussXCoords = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
        self.gaussZCoords = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
        self.gaussWeights = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
        self.gaussCosines = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
        self.gaussSines = np.zeros((numElements, NUM_REFERENCE_ELEMENT_POINTS))
        for elemIdx in range(numElements):
            localKug = np.zeros((self.numPoints, NUM_POINTS_IN_ELEMENT))
            localKqg = np.zeros((self.numPoints, NUM_POINTS_IN_ELEMENT))
            elemConnection = self.connectionMatrix[elemIdx]
            elemX = self.pointXCoords[elemConnection]
            elemZ = self.pointZCoords[elemConnection]
            for refIdx in range(NUM_REFERENCE_ELEMENT_POINTS):
                gaussX = np.sum(self.shapeFunctions[:, refIdx] * elemX)
                gaussZ = np.sum(self.shapeFunctions[:, refIdx] * elemZ)
                gaussDx = np.sum(self.shapeDerivatives[:, refIdx] * elemX)
                gaussDz = np.sum(self.shapeDerivatives[:, refIdx] * elemZ)
                deltaS = np.sqrt(gaussDx**2 + gaussDz**2)
                cosine = gaussDx / deltaS
                sine = gaussDz / deltaS
                distancesX = gaussX - self.colocationXCoords[:self.numPoints]
                distancesZ = gaussZ - self.colocationZCoords[:self.numPoints]
                radii = np.sqrt(distancesX**2 + distancesZ**2)
                greensFunc = -np.log(radii) / (2 * np.pi)
                greensFuncNormal = -(sine * distancesX - cosine * distancesZ) / (2 * np.pi * radii**2)
                for ptIdx in range(NUM_POINTS_IN_ELEMENT):
                    shapeVal = self.shapeFunctions[ptIdx, refIdx]
                    weight = deltaS * self.referenceWeights[refIdx]
                    localKug[:, ptIdx] += shapeVal * greensFunc * weight
                    localKqg[:, ptIdx] += shapeVal * greensFuncNormal * weight
                self.gaussXCoords[elemIdx, refIdx] = gaussX
                self.gaussZCoords[elemIdx, refIdx] = gaussZ
                self.gaussWeights[elemIdx, refIdx] = deltaS * self.referenceWeights[refIdx]
                self.gaussCosines[elemIdx, refIdx] = cosine
                self.gaussSines[elemIdx, refIdx] = sine
            for ptIdx in range(NUM_POINTS_IN_ELEMENT):
                connectionIdx = elemConnection[ptIdx]
                if connectionIdx < self.numPoints:
                    self.kugMatrix[:, connectionIdx] += localKug[:, ptIdx]
                    self.kqgMatrix[:, connectionIdx] += localKqg[:, ptIdx]

    def computeRhs(self, orientationVector, velocityVector, accelerationVector):
        orientationVector = np.array(orientationVector, dtype=float)
        velocityVector = np.array(velocityVector, dtype=float)
        accelerationVector = np.array(accelerationVector, dtype=float)
        U_inf = np.dot(velocityVector, orientationVector)
        velocity_perp = velocityVector - U_inf * orientationVector
        perp_vector = np.array([-orientationVector[1], orientationVector[0]])
        W_inf = np.dot(velocity_perp, perp_vector)
        U_inf_t = np.dot(accelerationVector, orientationVector)
        acceleration_perp = accelerationVector - U_inf_t * orientationVector
        W_inf_t = np.dot(acceleration_perp, perp_vector)
        if self.hasTrailingEdge:
            rhs = (U_inf * self.normalX + W_inf * self.normalZ)
            rhsT = (U_inf_t * self.normalX + W_inf_t * self.normalZ)
        else:
            rhs = -(U_inf * self.normalX + W_inf * self.normalZ)
            rhsT = -(U_inf_t * self.normalX + W_inf_t * self.normalZ)
        localVelocityVector = [U_inf, W_inf]
        localAccelerationVector = [U_inf_t, W_inf_t]
        return localVelocityVector, rhs, localAccelerationVector, rhsT

    def solveForPotential(self, rhs):
        phi = np.linalg.solve(self.kqgMatrix, rhs)
        return phi

    def solveForPotentialT(self, rhs):
        phiT = np.linalg.solve(self.kqgMatrix, rhs)
        phiT = self.kugMatrix @ phiT
        return phiT

    def solveForPotentialWithKJCondition(self, localVelocityVector, rhs, localAccelerationVector, rhsT):
        rhs = np.dot(self.kugMatrix, rhs)
        lowerPointIndex = (self.numPoints // 2) + 1
        rhs[lowerPointIndex] = 0
        rhsKJ = -2 * localVelocityVector[1] * self.normalX[lowerPointIndex]
        rhs = np.concatenate([rhs, [rhsKJ]])
#         print(np.shape(self.modifiedKqgMatrix))
        solution = np.linalg.solve(self.modifiedKqgMatrix, rhs)
        phi = -solution[:self.numPoints]
        gamma = solution[self.numPoints]
        rhsT[lowerPointIndex] = 0
        rhsTKJ = -2 * localAccelerationVector[1] * self.normalX[lowerPointIndex]
        rhsT = np.concatenate([rhsT, [rhsTKJ]])
        solution = np.linalg.solve(self.modifiedKqgMatrix, rhsT)
        phiT = solution[:self.numPoints]
        gammaT = solution[self.numPoints]
        return phi, gamma, phiT

    def computeTangentialTotalVelocity(self, perturbationPhi, localVelocityVector):
        U_inf = localVelocityVector[0]
        W_inf = localVelocityVector[1]
        totalPhi = perturbationPhi + localVelocityVector[0] * self.pointXCoords + localVelocityVector[1] * self.pointZCoords
        tangentialPertVel = np.zeros(self.numPoints)
        tangentialTotalVel = np.zeros(self.numPoints)
        intrinsicCoordinates = [-1, 0, 1]
        shapeFunctionsIntrinsic = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        shapeFunctionDerivatives = [[-1.5, -0.5, 0.5], [2, 0, -2], [-0.5, 0.5, 1.5]]
        numElements = len(self.connectionMatrix)
        for elementIndex in range(numElements):
            elementConnectionMatrix = self.connectionMatrix[elementIndex, :]
            elementPoints = elementConnectionMatrix
            elementXCoordinates = self.pointXCoords[elementPoints]
            elementZCoordinates = self.pointZCoords[elementPoints]
            elementWeights = []
            if self.hasTrailingEdge and elementIndex == numElements//2 - 1:
                elementWeights = np.array([1, 1, 0.5])
            elif self.hasTrailingEdge and elementIndex == numElements//2:
                elementWeights = np.array([0.5, 1, 1])
            else:
                elementWeights = np.array([0.5, 1, 0.5])
            for pointIndex in range(NUM_POINTS_IN_ELEMENT):
                if pointIndex == 0:
                    shapeFunctionsIntrinsic = [1, 0, 0]
                    shapeFunctionDerivatives = [-1.5, 2, -0.5]
                elif pointIndex == 1:
                    shapeFunctionsIntrinsic = [0, 1, 0]
                    shapeFunctionDerivatives = [-0.5, 0, 0.5]
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
                tangentialPertVelLocal = sum(shapeFunctionDerivatives * perturbationPhi[elementConnectionMatrix[:]])/pointDeltaArclength
                tangentialTotalVelLocal = tangentialPertVelLocal + U_inf*pointCosine + W_inf*pointSine
                tangentialPertVel[elementConnectionMatrix[pointIndex]] = tangentialPertVel[elementConnectionMatrix[pointIndex]] + elementWeights[pointIndex] * tangentialPertVelLocal
                tangentialTotalVel[elementConnectionMatrix[pointIndex]] = tangentialTotalVel[elementConnectionMatrix[pointIndex]] + elementWeights[pointIndex] * tangentialTotalVelLocal
        return tangentialPertVel, tangentialTotalVel

    def addVortexContribution(self, phi, phiT, tangentialTotalVelocity, gamma):
        phi += gamma * self.vortexPhi
        phiT += gamma**2 * self.vortexPhiT
        tangentialTotalVelocity[:self.numPoints//2+1] += (gamma * self.vortexVelocityX[:self.numPoints//2+1] * -self.normalZ[:self.numPoints//2+1]) + (gamma * self.vortexVelocityZ[:self.numPoints//2+1] * self.normalX[:self.numPoints//2+1])
        tangentialTotalVelocity[self.numPoints//2+1:] += (gamma * self.vortexVelocityX[self.numPoints//2+1:] * -self.normalZ[self.numPoints//2+1:]) + (gamma * self.vortexVelocityZ[self.numPoints//2+1:] * self.normalX[self.numPoints//2+1:])
        return phi, phiT, tangentialTotalVelocity

    def computeForce(self, velocityVector, phi, phiT, tangentialTotalVel):
        U_mag_squared = velocityVector[0]**2 + velocityVector[1]**2
        forceZ = 0
        forceX = 0
        dynamicPressures = []
        numElements = len(self.connectionMatrix)
        for elementIndex in range(numElements):
            for gaussPointIndex in range(NUM_REFERENCE_ELEMENT_POINTS):
                tangentialVelAtGaussPoint = sum(self.shapeFunctions[:, gaussPointIndex] * tangentialTotalVel[self.connectionMatrix[elementIndex]])
                phiTAtGaussPoint = sum(self.shapeFunctions[:, gaussPointIndex] * phiT[self.connectionMatrix[elementIndex]])
                dynamicPressure = 0.5*RHO*(U_mag_squared - tangentialVelAtGaussPoint**2) + RHO*phiTAtGaussPoint
                dynamicPressures.append(dynamicPressure)
                forceZ = forceZ - dynamicPressure * self.gaussCosines[elementIndex, gaussPointIndex] * self.gaussWeights[elementIndex, gaussPointIndex]
                forceX = forceX - dynamicPressure * self.gaussSines[elementIndex, gaussPointIndex] * self.gaussWeights[elementIndex, gaussPointIndex]
        forceVector = [forceX, forceZ]
        return forceVector

    def projectForceVector(self, orientationVector, forceVector):
        orient = np.array(orientationVector, dtype=float)
        force = np.array(forceVector, dtype=float)
        norm = np.sqrt(orient @ orient)
        orient = orient / norm
        force_along = force @ orient
        perp_vector = np.array([-orient[1], orient[0]])
        force_perp = force @ perp_vector
        return [force_along, force_perp]

    def computeLocalVector(self, vector, orientationVector):
        orient = np.array(orientationVector, dtype=float)
        norm = np.sqrt(orient @ orient)
        if norm == 0:
            return np.array([0, 0])
        localXAxis = orient / norm
        localZAxis = np.array([-localXAxis[1], localXAxis[0]])
        return np.array([np.dot(vector, localXAxis), np.dot(vector, localZAxis)])

    # Restore the original computeForceFromFlow logic
    def computeForceFromFlow(self, orientationVector, velocityVector, accelerationVector):
        # Compute local vectors and right-hand sides
        localVelocityVector, rhs, localAccelerationVector, rhsT = self.computeRhs(
            orientationVector, velocityVector, accelerationVector
        )
        
        # Solve for potentials and circulation
        if self.hasTrailingEdge:
            phi, gamma, phiT = self.solveForPotentialWithKJCondition(
                localVelocityVector, rhs, localAccelerationVector, rhsT
            )
        else:
            phi = self.solveForPotential(rhs)
            phiT = self.solveForPotentialT(rhsT)
            gamma = 0  # No vortex for circles

        # Compute tangential velocities
        tangentialPertVel, tangentialTotalVel = self.computeTangentialTotalVelocity(
            phi, localVelocityVector
        )

        # Add vortex contribution if applicable
        if self.hasTrailingEdge:
            phi, phiT, tangentialTotalVel = self.addVortexContribution(
                phi, phiT, tangentialTotalVel, gamma
            )

        # Compute force using dynamic pressure
        localForceVector = self.computeForce(localVelocityVector, phi, phiT, tangentialTotalVel)


        # Store results for rendering
        self.localVelocityVector = localVelocityVector
        self.localAccelerationVector = localAccelerationVector
        self.tangentialTotalVelocity = tangentialTotalVel
        self.localForceVector = localForceVector
        
        forceVector = self.projectForceVector(orientationVector, localForceVector)
        
#         print("localVelocityVector", localVelocityVector)
#         print("acceleration vector", accelerationVector)
#         print("local acceleration vector", localAccelerationVector)
#         print("orientation vector", orientationVector)
#         print("points x", self.pointXCoords)
#         print("points z", self.pointZCoords)
#         print("colocation x", self.colocationXCoords)
#         print("colocation z", self.colocationZCoords)
#         print("local force vector", localForceVector)
#         print("force vector", forceVector)
#         self.plotGeometry("test.png")

        return forceVector

    # Keep the plotting methods unchanged
    def plotGeometry(self, filename):
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(self.pointXCoords, self.pointZCoords, label="points", s=5)
        plt.scatter(self.colocationXCoords, self.colocationZCoords, label="colocation points", s=5)
        for element in self.connectionMatrix:
            plt.plot(self.pointXCoords[element], self.pointZCoords[element])
        plt.title("num points:" + str(len(self.pointXCoords)))
        plt.legend()
        plt.savefig(os.path.join('graphs', filename))
#         plt.show()
        plt.close()

    def plotNormals(self, filename):
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(self.pointXCoords, self.pointZCoords, label="points", s=5)
        plt.scatter(self.colocationXCoords, self.colocationZCoords, label="colocation points", s=5)
        for element in self.connectionMatrix:
            plt.plot(self.pointXCoords[element], self.pointZCoords[element])
        for i in range(len(self.pointXCoords)):
            plt.arrow(self.pointXCoords[i], self.pointZCoords[i],
                      self.normalX[i] * 0.1, self.normalZ[i] * 0.1,
                      head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal Vectors' if i == 0 else "")
        plt.title("normal vectors")
        plt.legend()
        plt.savefig(os.path.join('graphs', filename))
        plt.close()

    def plotForces(self, filename, localVelocityVector, tangentialTotalVelocity, forceVector):
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(self.pointXCoords, self.pointZCoords, label="points", s=5)
        plt.scatter(self.colocationXCoords, self.colocationZCoords, label="colocation points", s=5)
        for element in self.connectionMatrix:
            plt.plot(self.pointXCoords[element], self.pointZCoords[element])
        scale = 0.1
        for i in range(len(self.pointXCoords)):
            plt.arrow(self.pointXCoords[i], self.pointZCoords[i],
                      tangentialTotalVelocity[i] * -self.normalZ[i] * scale, tangentialTotalVelocity[i] * self.normalX[i] * scale,
                      head_width=0.005, head_length=0.005, fc='green', ec='green', label='Tangential Velocity Vectors' if i == 0 else "")
        plt.arrow(min(self.pointXCoords) - 0.1, 0,
                  localVelocityVector[0] * scale/2, localVelocityVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity' if i == 0 else "")
        plt.arrow(min(self.pointXCoords) - 0.1, -.1,
                  localVelocityVector[0] * scale/2, localVelocityVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity' if i == 0 else "")
        plt.arrow(min(self.pointXCoords) - 0.1, .1,
                  localVelocityVector[0] * scale/2, localVelocityVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity' if i == 0 else "")
        plt.arrow(min(self.pointXCoords) - 0.1, .2,
                  localVelocityVector[0] * scale/2, localVelocityVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity' if i == 0 else "")
        plt.arrow(self.pointXCoords[self.numPoints//4], 0,
                  forceVector[0] * 0.01, forceVector[1] * 0.01,
                  head_width=0.00001, head_length=0.00001, fc='pink', ec='pink', label='Force Vector' if i == 0 else "")
        plt.title("Local Velocity Frame")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plotPotential(self, filename, phi):
        plt.figure()
        plt.plot(self.pointXCoords[:self.numPoints//2+1], phi[:self.numPoints//2+1], label="upper")
        plt.plot(self.pointXCoords[self.numPoints//2+1:], phi[self.numPoints//2+1:], label="lower")
        plt.legend()
        plt.xlabel("equal")
        plt.ylabel("Perturbation Potential")
        plt.title("Perturbation Potential Along Boundary")
        plt.savefig(os.path.join('graphs', filename))
        plt.close()

    def plotVelocity(self, filename, tangentialTotalVelocity):
        plt.figure()
        plt.plot(self.pointXCoords[:self.numPoints//2+1], tangentialTotalVelocity[:self.numPoints//2+1], label="upper")
        plt.plot(self.pointXCoords[self.numPoints//2+1:], tangentialTotalVelocity[self.numPoints//2+1:], label="lower")
        plt.legend()
        plt.xlabel("equal")
        plt.ylabel("Velocity")
        plt.title("Tangential Total Velocity Along Boundary")
        plt.savefig(os.path.join('graphs', filename))
        plt.close()

CIRCLE_COLOCATION_SCALING_FACTOR = 0.95        
class Circle:
    def __init__(self, radius, mass):
        self.numPoints = 300
        self.radius = radius
        self.mass = mass
        theta = np.linspace(0 + ((np.pi)/self.numPoints), 2 * np.pi - ((np.pi)/self.numPoints), self.numPoints, endpoint=True)
        pointXCoords = self.radius * np.cos(theta)
        pointZCoords = self.radius * np.sin(theta)
        colocationXCoords = CIRCLE_COLOCATION_SCALING_FACTOR * pointXCoords
        colocationZCoords = CIRCLE_COLOCATION_SCALING_FACTOR * pointZCoords
        self.geometry = Geometry(pointXCoords, pointZCoords, colocationXCoords, colocationZCoords, hasTrailingEdge=False)
        self.hasTrailingEdge = False  # Explicitly set
        self.generateElementMatrices()
        
    def generateElementMatrices(self):
        self.numElements = ((self.numPoints - 1) // (NUM_POINTS_IN_ELEMENT - 1)) + 1
        connectionMatrix = np.zeros((self.numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
        assemblyMatrix = np.zeros((self.numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
        
        for i in range(self.numElements):
            start = i * (NUM_POINTS_IN_ELEMENT - 1)
            connectionMatrix[i] = [(start + j) % self.numPoints for j in range(NUM_POINTS_IN_ELEMENT)]
            assemblyMatrix[i] = [(start + j) % self.numPoints for j in range(NUM_POINTS_IN_ELEMENT)]
        
        self.geometry.setElementMatrices(connectionMatrix, assemblyMatrix)



class Foil:
    def __init__(self, foilType=None, chordLength=0.203, numPoints=300, points=None, mass=300):
        self.foilType = foilType
        self.chordLength = chordLength
        self.numPoints = numPoints + 1  # Account for double-counted trailing edge
        self.mass = mass
        if points:
            nacaXRaw, nacaZRaw = points
            nacaXRaw = np.array(nacaXRaw) * self.chordLength
            nacaZRaw = np.array(nacaZRaw) * self.chordLength
        else:
            nacaXRaw = np.arange(0, 1 + NACA_FOIL_RESOLUTION, NACA_FOIL_RESOLUTION)
            nacaZRaw = self.generateNacaPoints(nacaXRaw)
        numPointsNoseToTip = int(self.numPoints // 2 + 1)  # Convert to integer
        nacaNoseToTipXCoords, nacaNoseToTipZCoords = self.computeArclengthInterpolation(nacaXRaw, nacaZRaw, numPointsNoseToTip)
        nacaNoseToTipXCoords = nacaNoseToTipXCoords * self.chordLength
        nacaNoseToTipZCoords = nacaNoseToTipZCoords * self.chordLength
        pointXCoords = np.concatenate((nacaNoseToTipXCoords, nacaNoseToTipXCoords[-1:0:-1]))
        pointZCoords = np.concatenate((nacaNoseToTipZCoords, -nacaNoseToTipZCoords[-1:0:-1]))
        colocationXCoords, colocationZCoords = self.computeColocationPoints(
            nacaNoseToTipXCoords, nacaNoseToTipZCoords)
        self.geometry = Geometry(pointXCoords, pointZCoords, colocationXCoords, colocationZCoords)
        self.generateElementMatrices()
        self.hasTrailingEdge = True  # Explicitly set for zooming logic

    def generateNacaPoints(self, xCoords):
        if self.foilType == NACA_0012_FOIL_TYPE:
            C0 = 0.594689181
            C1 = 0.298222773
            C2 = -0.127125232
            C3 = -0.357907906
            C4 = 0.291984971
            C5 = -0.105174606
            return C0*(C1*np.sqrt(xCoords) + C2*xCoords + C3*xCoords**2 + C4*xCoords**3 + C5*xCoords**4)
        return np.zeros_like(xCoords)  # Fallback for custom points

    def computeArclengthInterpolation(self, xCoords, zCoords, numPoints):
        cumulativeArclengths = np.zeros(len(xCoords))
        for i in range(1, len(xCoords)):
            cumulativeArclengths[i] = cumulativeArclengths[i-1] + np.sqrt(
                (xCoords[i] - xCoords[i-1])**2 + (zCoords[i] - zCoords[i-1])**2)
        arclengthMax = cumulativeArclengths[-1]
        pointArclengths = np.linspace(0, arclengthMax, numPoints)
        interpolateX = interp1d(cumulativeArclengths, xCoords, kind='cubic',
                                bounds_error=False, fill_value=(xCoords[0], xCoords[-1]))
        interpolateZ = interp1d(cumulativeArclengths, zCoords, kind='cubic',
                                bounds_error=False, fill_value=(zCoords[0], zCoords[-1]))
        xPoints = interpolateX(pointArclengths)
        zPoints = interpolateZ(pointArclengths) if self.foilType is None else self.generateNacaPoints(xPoints)
        return xPoints, zPoints

    def computeColocationPoints(self, xCoords, zCoords):
        scaleX = COLOCATION_SCALING_FACTOR_X
        scaleZ = COLOCATION_SCALING_FACTOR_Z
        chordLength = max(xCoords) - min(xCoords)
        midPointShift = 0.5 * (1 - scaleX) * chordLength
        colocationXUpper = scaleX * xCoords + midPointShift
        colocationXLower = scaleX * xCoords[-1:0:-1] + midPointShift
        colocationZUpper = scaleZ * zCoords
        colocationZLower = -scaleZ * zCoords[-1:0:-1]
        return (np.concatenate((colocationXUpper, colocationXLower)),
                np.concatenate((colocationZUpper, colocationZLower)))

    def generateElementMatrices(self):
        self.numElements = int((self.numPoints - 1) // (NUM_POINTS_IN_ELEMENT - 1))
        connectionMatrix = np.zeros((self.numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
        assemblyMatrix = np.zeros((self.numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
        halfElements = self.numElements // 2
        for i in range(halfElements):
            start = i * 2
            connectionMatrix[i] = np.arange(start, start + NUM_POINTS_IN_ELEMENT)
            connectionMatrix[i + halfElements] = np.arange(
                start + (self.numPoints // 2) + 1, start + (self.numPoints // 2) + NUM_POINTS_IN_ELEMENT + 1)
            assemblyMatrix[i] = np.arange(start, start + NUM_POINTS_IN_ELEMENT)
            assemblyMatrix[i + halfElements] = np.arange(
                start + self.numPoints // 2, start + self.numPoints // 2 + NUM_POINTS_IN_ELEMENT)
        # Connect the last element to the nose point
        connectionMatrix[-1][-1] = 0
        assemblyMatrix[-1][-1] = 0
        self.geometry.setElementMatrices(connectionMatrix, assemblyMatrix)