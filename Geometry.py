from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os

RHO = 1000
NUM_REFERENCE_ELEMENT_POINTS = 80
NUM_POINTS_IN_ELEMENT = 3
class Geometry:
    def __init__(self, pointXCoords, pointZCoords, colocationXCoords, colocationZCoords, hasTrailingEdge=True):
        self.pointXCoords = pointXCoords
        self.pointZCoords = pointZCoords
        self.colocationXCoords = colocationXCoords
        self.colocationZCoords = colocationZCoords
        self.numPoints = len(self.pointXCoords)
        self.hasTrailingEdge = hasTrailingEdge
        self.generateNormals()
        self.generateReferenceElementData()
        
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
        if(self.hasTrailingEdge):
            self.generateVortex()
            
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
        # Initialize output arrays
        vortexVelocityX = np.zeros(self.numPoints)  # Discrete x-velocity
        vortexVelocityZ = np.zeros(self.numPoints)  # Discrete z-velocity
        vortexVelocityXIntegrated = np.zeros(self.numPoints)  # Integrated x-velocity (uvtxig equivalent)
        vortexVelocityZIntegrated = np.zeros(self.numPoints)  # Integrated z-velocity (vvtxig equivalent)
        vortexVelocityNormal = np.zeros(self.numPoints)
        vortexPhi = np.zeros(self.numPoints)
        vortexPhiT = np.zeros(self.numPoints)
        thetas = []
        # Compute velocities for each point
        for pointIndex in range(self.numPoints):
            theta = np.zeros(self.numVortexPoints)
            distanceX = self.pointXCoords[pointIndex] - self.vortexLineXCoordinates
            distanceZ = self.pointZCoords[pointIndex] - self.vortexLineZCoordinates
            distanceMagnitude = np.sqrt(distanceX**2 + distanceZ**2)
            
            # Avoid division by zero
            distanceMagnitude[distanceMagnitude < 1e-10] = 1e-10
            
            # Compute theta for all vortex points
            theta = np.arctan2(distanceZ, distanceX)
            if(max(theta) < 0):
                theta += 2*np.pi
            thetas.append(theta)
            vortexPhi[pointIndex] = ((1 / (2 * np.pi * self.numVortexPoints)) * np.sum(theta))
            vortexPhiT[pointIndex] = (1/(4 * np.pi**2 * np.sum(distanceMagnitude**2)))
    #         print(vortexPhi[pointIndex])
    #         quit()
            # Discrete velocities (non-integrated case)
            if not integrated:
                vortexVelocityX[pointIndex] = (-1 / (2 * np.pi * self.numVortexPoints)) * np.sum(distanceZ / (distanceMagnitude**2))
                vortexVelocityZ[pointIndex] = (1 / (2 * np.pi * self.numVortexPoints)) * np.sum(distanceX / (distanceMagnitude**2))
                vortexVelocityNormal[pointIndex] = (vortexVelocityX[pointIndex] * self.normalX[pointIndex]) + \
                                                   (vortexVelocityZ[pointIndex] * self.normalZ[pointIndex])
    
            # Integrated velocities (always compute, used when integrated=True)
            vortexVelocityXIntegrated[pointIndex] = (-1 / (2 * np.pi * self.vortexLineLength)) * (theta[-1] - theta[0])
            
            # Z-component: Default integrated case
            if abs(self.pointZCoords[pointIndex]) < 1e-6:  # Leading or trailing edge (z ≈ 0)
                vortexVelocityZIntegrated[pointIndex] = (1 / (2 * np.pi * self.vortexLineLength)) * np.log(
                    abs(self.pointXCoords[pointIndex] - self.vortexLineXCoordinates[0]) / 
                    abs(self.pointXCoords[pointIndex] - self.vortexLineXCoordinates[-1])
                )
            else:
                sin_theta_start = np.sin(theta[0])
                sin_theta_end = np.sin(theta[-1])
                if abs(sin_theta_start) < 1e-10 or abs(sin_theta_end) < 1e-10:
                    vortexVelocityZIntegrated[pointIndex] = 0  # Avoid log(0)
                else:
                    vortexVelocityZIntegrated[pointIndex] = (1 / (2 * np.pi * self.vortexLineLength)) * np.log(
                        abs(sin_theta_end / sin_theta_start)
                    )
    
            # Use integrated velocities if flag is True
            if integrated:
                vortexVelocityX[pointIndex] = vortexVelocityXIntegrated[pointIndex]
                vortexVelocityZ[pointIndex] = vortexVelocityZIntegrated[pointIndex]
                vortexVelocityNormal[pointIndex] = -((vortexVelocityX[pointIndex] * self.normalX[pointIndex]) + \
                                                   (vortexVelocityZ[pointIndex] * self.normalZ[pointIndex]))
    
    #     print("index", numPoints//2 + 1)
    #     print("vortexPhi", vortexPhi)
        vortexPhi[self.numPoints//2 + 1] = vortexPhi[self.numPoints//2]
        vortexPhiT[self.numPoints//2 + 1] = vortexPhiT[self.numPoints//2]
        self.vortexPhi = vortexPhi
        self.vortexPhiT = vortexPhiT
        self.vortexVelocityX = vortexVelocityX
        self.vortexVelocityZ = vortexVelocityZ
        self.vortexVelocityNormal = vortexVelocityNormal
#         print("vortexVelocityNormal", vortexVelocityNormal)

            
    def generateModifiedKqgMatrix(self):
        upperPointIndex = (self.numPoints // 2)
        lowerPointIndex = (self.numPoints // 2) + 1
        vortexInfluenceMatrix = np.dot(self.kugMatrix, self.vortexVelocityNormal)
        vortexInfluenceMatrix[upperPointIndex] = 0
        vortexInfluenceMatrix[lowerPointIndex] = 0
        
        # Continuity condition
        modifiedKqgMatrix = self.kqgMatrix.copy()
        maxKqg = np.max(np.abs(np.diag(modifiedKqgMatrix)))
        modifiedKqgMatrix[lowerPointIndex, :] = 0
        modifiedKqgMatrix[lowerPointIndex, lowerPointIndex] = maxKqg
        modifiedKqgMatrix[lowerPointIndex, upperPointIndex] = -maxKqg
    
        # Debug: Print rhs
    
        # Shape function derivatives at xi = 1, -1
        shapeFunctionDerivativesOne = np.array([0.5, -2, 1.5])  # xi = 1
        shapeFunctionDerivativesMinusOne = np.array([-1.5, 2, -0.5])  # xi = -1
    
        numElements = len(self.connectionMatrix)
        # Element indices (trailing edge elements)
        upperElementIndex = (numElements // 2) - 1  # Last element of upper surface
        lowerElementIndex = (numElements // 2)  # First element of lower surface
    
        # Debug: Print element indices
    
        # KJ row
        kjRow = np.zeros(self.numPoints)
    
        # Upper element (end at xi = 1)
        upperElementPointIndexes = self.connectionMatrix[upperElementIndex]
        upperElementXCoordinates = np.sum(shapeFunctionDerivativesOne * self.pointXCoords[upperElementPointIndexes])
        upperElementZCoordinates = np.sum(shapeFunctionDerivativesOne * self.pointZCoords[upperElementPointIndexes])
        upperElementDeltaArclength = np.sqrt(upperElementXCoordinates**2 + upperElementZCoordinates**2)
        upperElementTangentialVelocity = shapeFunctionDerivativesOne / upperElementDeltaArclength
        kjRow[upperElementPointIndexes] = upperElementTangentialVelocity
    
        # Lower element (start at xi = -1)
        lowerElementPointIndexes = self.connectionMatrix[lowerElementIndex]
        lowerElementXCoordinates = np.sum(shapeFunctionDerivativesMinusOne * self.pointXCoords[lowerElementPointIndexes])
        lowerElementZCoordinates = np.sum(shapeFunctionDerivativesMinusOne * self.pointZCoords[lowerElementPointIndexes])
        lowerElementDeltaArclength = np.sqrt(lowerElementXCoordinates**2 + lowerElementZCoordinates**2)
        lowerElementTangentialVelocity = shapeFunctionDerivativesMinusOne / lowerElementDeltaArclength
        kjRow[lowerElementPointIndexes] = lowerElementTangentialVelocity

    
        # Build system matrix with KJ condition
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
#         Project the velocity vector and acceleration vector onto the oriantation vector
#       To calculate U_inf W_inf U_inf_t and W_inf_t in foil local coordinates
        orientationVector = np.array(orientationVector, dtype=float)
        velocityVector = np.array(velocityVector, dtype=float)
        accelerationVector = np.array(accelerationVector, dtype=float)
        # Project velocity onto orientation vector (U_inf is along orientation)
        U_inf = np.dot(velocityVector, orientationVector)  # Component along orientation
    
        # Compute perpendicular velocity vector
        velocity_perp = velocityVector - U_inf * orientationVector
    
        # W_inf is the signed component of velocity perpendicular to orientation
        # Assuming a 2D system, we use the scalar projection onto a perpendicular vector
        # Define a perpendicular vector (e.g., in 2D, rotate orientation vector by 90 degrees)
        perp_vector = np.array([-orientationVector[1], orientationVector[0]])  # For 2D
        W_inf = np.dot(velocity_perp, perp_vector)  # Signed perpendicular component
    
        # Project acceleration onto orientation vector (U_inf_t is along orientation)
        U_inf_t = np.dot(accelerationVector, orientationVector)  # Acceleration along orientation
    
        # Compute perpendicular acceleration vector
        acceleration_perp = accelerationVector - U_inf_t * orientationVector
    
        # W_inf_t is the signed component of acceleration perpendicular to orientation
        W_inf_t = np.dot(acceleration_perp, perp_vector)  # Signed perpendicular acceleration
#         print("U_inf, W_inf", U_inf, W_inf, U_inf_t, W_inf_t)
#         Reverse velocity values to change to apparent current
#         U_inf = -U_inf
#         W_inf = -W_inf
#         U_inf_t = -U_inf_t
#         W_inf_t = -W_inf_t
        
        rhs = (U_inf * self.normalX + W_inf * self.normalZ)
        rhsT = (U_inf_t * self.normalX + W_inf_t * self.normalZ)

        
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
#         print("self.kugMatrix", self.kugMatrix)
#         print("self.kqgMatrix", self.modifiedKqgMatrix[:self.numPoints, :self.numPoints])
#         Verify kug done
#        Verify kqg its off
#       Verify vortex velocity normal good
#        Verify each modification of kqg
#         print("self.ModifiedKqgMatrix", self.modifiedKqgMatrix)
#        Verify rhs
#        Verify solution
#        Verify tangential velocities
#        Verify forces
        lowerPointIndex = (self.numPoints // 2) + 1
        rhs[lowerPointIndex] = 0
        rhsKJ = -2 * localVelocityVector[1] * self.normalX[lowerPointIndex]
        rhs = np.concatenate([rhs, [rhsKJ]])
#         print("self.modifiedKqgMatrix", self.modifiedKqgMatrix)
        
        solution = np.linalg.solve(self.modifiedKqgMatrix, rhs)
        phi = -solution[:self.numPoints]
        gamma = solution[self.numPoints]
        

#         phiT = self.solveForPotentialT(rhsT)
        rhsT[lowerPointIndex] = 0
        rhsTKJ = -2 * localAccelerationVector[1] * self.normalX[lowerPointIndex]
        rhsT = np.concatenate([rhsT, [rhsTKJ]])
# #         print("self.modifiedKqgMatrix", self.modifiedKqgMatrix)
#         
        solution = np.linalg.solve(self.modifiedKqgMatrix, rhsT)
        phiT = -solution[:self.numPoints]
        gammaT = solution[self.numPoints]
#         print(phiT)
        
        return phi, gamma, phiT
        
    def computeTangentialTotalVelocity(self, perturbationPhi, localVelocityVector):
        U_inf = localVelocityVector[0]
        W_inf = localVelocityVector[1]
#     (kugMatrix, phiNormal, vortexGamma, vortexPhi, xCoords, zCoords, numPoints, U_inf, W_inf):
        totalPhi = perturbationPhi + localVelocityVector[0] * self.pointXCoords + localVelocityVector[1] * self.pointZCoords
        tangentialPertVel = np.zeros(self.numPoints)
        tangentialTotalVel = np.zeros(self.numPoints)
        intrinsicCoordinates = [-1, 0, 1]
    #     shapeFunctions = funf1(intrinsicCoordinates, NUM_POINTS_IN_ELEMENT)
    #     shapeFunctionDerivatives = dfunf1(intrinsicCoordinates, NUM_POINTS_IN_ELEMENT)
        shapeFunctionsIntrinsic = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        shapeFunctionDerivatives = [[-1.5, -0.5, 0.5], [2, 0, -2], [-0.5, 0.5, 1.5]]
        numElements = len(self.connectionMatrix)
        for elementIndex in range(numElements):
            elementConnectionMatrix = self.connectionMatrix[elementIndex, :]
            elementPoints = elementConnectionMatrix
    #         print(elementPoints)
            elementXCoordinates = self.pointXCoords[elementPoints]
            elementZCoordinates = self.pointZCoords[elementPoints]
            elementWeights = []
            if(self.hasTrailingEdge and elementIndex == numElements//2 - 1):
                elementWeights = np.array([1, 1, 0.5])
            elif(self.hasTrailingEdge and elementIndex == numElements//2):
                elementWeights = np.array([0.5, 1, 1])
            else:
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
                tangentialTotalVelLocal = tangentialPertVelLocal + U_inf*pointCosine + W_inf*pointSine
    #             print("tangentialPertLocal", tangentialPertVelLocal)
    #             print("tangentialTotalLocal", tangentialTotalVelLocal)
                tangentialPertVel[elementConnectionMatrix[pointIndex]] = tangentialPertVel[elementConnectionMatrix[pointIndex]] + elementWeights[pointIndex] * tangentialPertVelLocal
                tangentialTotalVel[elementConnectionMatrix[pointIndex]] = tangentialTotalVel[elementConnectionMatrix[pointIndex]] + elementWeights[pointIndex] * tangentialTotalVelLocal

    

        return tangentialPertVel, tangentialTotalVel
        
        
#     TODO: Combime with calculateVortexVelocityNormal function
    def addVortexContribution(self, phi, phiT, tangentialTotalVelocity, gamma):
        phi += gamma * self.vortexPhi
        phiT += gamma**2 * self.vortexPhiT
        tangentialTotalVelocity[:self.numPoints//2+1] += (gamma * self.vortexVelocityX[:self.numPoints//2+1] * -self.normalZ[:self.numPoints//2+1]) + (gamma * self.vortexVelocityZ[:self.numPoints//2+1] * self.normalX[:self.numPoints//2+1])
        tangentialTotalVelocity[self.numPoints//2+1:] += (gamma * self.vortexVelocityX[self.numPoints//2+1:] * -self.normalZ[self.numPoints//2+1:]) + (gamma * self.vortexVelocityZ[self.numPoints//2+1:] * self.normalX[self.numPoints//2+1:])
        return phi, phiT, tangentialTotalVelocity

        
    def computeForce(self, velocityVector, phi, phiT, tangentialTotalVel):
#     perturbationPhi, phiT, tangentialVelX, tangentialVelZ, tangentialTotalVel, normalX, normalZ, ds, U_inf, W_inf, numPoints):
        # Compute velocity magnitude
#         velocityMag = (U_inf + tangentialVelX)**2 + (W_inf + tangentialVelZ)**2
    
        # Compute total pressure for forces (used for added mass, lift, and drag)
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
        """
        Project forceVector onto orientationVector and its perpendicular vector.
        Returns a 2D vector where the first component is the force along the orientation
        vector (preserving sign) and the second component is the force perpendicular to it
        (preserving sign).
        
        Args:
            orientationVector (list): 2D vector [x, y] defining the foil's orientation.
            forceVector (list): 2D force vector [fx, fy] in global coordinates.
        
        Returns:
            list: [force_along, force_perp] in foil local coordinates.
        """
        import numpy as np
        
        # Convert inputs to NumPy arrays
        orient = np.array(orientationVector, dtype=float)
        force = np.array(forceVector, dtype=float)
        
        # Normalize orientation vector
        norm = np.sqrt(orient @ orient)
        orient = orient / norm
        
        # Compute force along orientation vector (scalar projection)
        force_along = force @ orient
        
        # Compute perpendicular vector (rotate orientation vector 90 degrees counterclockwise)
        perp_vector = np.array([-orient[1], orient[0]])
        
        # Compute force perpendicular to orientation vector (signed projection)
        force_perp = force @ perp_vector
        
        return [force_along, force_perp]
    
    def computeForceFromFlow(self, orientationVector, velocityVector, accelerationVector):
        localVelocityVector, rhs, localAccelerationVector, rhsT = self.computeRhs(orientationVector, velocityVector, accelerationVector)
#         print(localVelocityVector, rhs)
        print("acceleration vector", accelerationVector)
        print("local acceleration vector", localAccelerationVector)
        if(self.hasTrailingEdge):
            phi, gamma, phiT = self.solveForPotentialWithKJCondition(localVelocityVector, rhs, localAccelerationVector, rhsT)
#             print("phi, gamma", phi, gamma)
#             quit()
        else:
            phi = self.solveForPotential(rhs)
            phiT = self.solveForPotentialT(rhsT)
            
#         print(phiT)
#         print(phi)
#         quit()
#         quit()
#         quit()
        tangentialPertVelocity, tangentialTotalVelocity = self.computeTangentialTotalVelocity(phi, localVelocityVector)

#         print("tangential total velocity", tangentialTotalVelocity)
#         quit()
        if(self.hasTrailingEdge):
            phi, phiT, tangentialTotalVelocity = self.addVortexContribution(phi, phiT, tangentialTotalVelocity, gamma)
        
        localForceVector = self.computeForce(localVelocityVector, phi, phiT, tangentialTotalVelocity)
        
#         Set as instance variables for making a local movie
        self.localVelocityVector = localVelocityVector
        self.tangentialTotalVelocity = tangentialTotalVelocity
        self.localForceVector = localForceVector
        
        self.plotPotential("object_potential.png", phi)
        self.plotVelocity("object_velocity.png", tangentialTotalVelocity)
#         print("gamma", gamma)
        U_infinity = np.sqrt(localVelocityVector[0]**2 + localVelocityVector[1]**2)
# # # #             lift = RHO * U_infinity * Gamma
        dynamicPressure = 0.5 * RHO * U_infinity**2
# # # #             foilLiftCoeff = lift / (dynamicPressure * NACA_CHORD_LENGTH) if dynamicPressure != 0 else 0
        foilLiftCoeff = -2 * gamma / (U_infinity * 0.203)
        liftPerUnitSpan = foilLiftCoeff * dynamicPressure * 0.203
#             print("GAMMA: ", Gamma)

        print("KUTTA LIFT FORCE", liftPerUnitSpan)
#         print("Local Force Vector", localForceVector)
#         print(orientationVector)
        forceVector = self.projectForceVector(orientationVector, localForceVector)
#         forceVector = localForceVector
#         print("local force vector", localForceVector)
#         project forceVector onto orientationVector
        return forceVector
        
#         tangentialVortexVelocity = self.computeTangentialVortexVelocity()
    
    
    
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
        plt.show()
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
    def __init__(self, radius, numPoints):
        self.numPoints = numPoints
        self.radius = radius
        theta = np.linspace(0 + ((np.pi)/self.numPoints), 2 * np.pi - ((np.pi)/self.numPoints), self.numPoints, endpoint=True)
        pointXCoords = self.radius * np.cos(theta)
        pointZCoords = self.radius * np.sin(theta)
        colocationXCoords = CIRCLE_COLOCATION_SCALING_FACTOR * pointXCoords
        colocationZCoords = CIRCLE_COLOCATION_SCALING_FACTOR * pointZCoords
        self.geometry = Geometry(pointXCoords, pointZCoords, colocationXCoords, colocationZCoords, hasTrailingEdge=False)
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




NACA_FOIL_RESOLUTION = 0.001
NACA_0012_FOIL_TYPE = "0012"
COLOCATION_SCALING_FACTOR_X = 0.995
COLOCATION_SCALING_FACTOR_Z = 0.95
class Foil:
    def __init__(self, foilType, chordLength, numPointsWithoutDoublePoint):
        self.foilType = foilType
        self.chordLength = chordLength
        self.numPoints = numPointsWithoutDoublePoint + 1
        
        nacaXRaw = np.arange(0, 1 + NACA_FOIL_RESOLUTION, NACA_FOIL_RESOLUTION)
        nacaZRaw = self.generateNacaPoints(nacaXRaw)
        numPointsNoseToTip = self.numPoints // 2 + 1
        nacaNoseToTipXCoords, nacaNoseToTipZCoords = self.computeArclengthInterpolation(nacaXRaw, nacaZRaw, numPointsNoseToTip)
        nacaNoseToTipXCoords = nacaNoseToTipXCoords * self.chordLength
        nacaNoseToTipZCoords = nacaNoseToTipZCoords * self.chordLength
#         Create the points to start at the nose, go to the tip,
#           DOUBLE COUNT the node on the tip. 
#           then concatenate the lower side, flipping the z coordinate and ending at the last but one point
        pointXCoords = np.concatenate((nacaNoseToTipXCoords, nacaNoseToTipXCoords[-1:0:-1]))
        pointZCoords = np.concatenate((nacaNoseToTipZCoords, -nacaNoseToTipZCoords[-1:0:-1]))
        colocationXCoords, colocationZCoords = self.computeColocationPoints(
            nacaNoseToTipXCoords, nacaNoseToTipZCoords, self.chordLength)
        self.geometry = Geometry(pointXCoords, pointZCoords, colocationXCoords, colocationZCoords)
        self.generateElementMatrices()
            
            
        
                
    def generateNacaPoints(self, xCoords):
        if(self.foilType == NACA_0012_FOIL_TYPE):
            C0 = 0.594689181
            C1 = 0.298222773
            C2 = -0.127125232
            C3 = -0.357907906
            C4 = 0.291984971
            C5 = -0.105174606
            return C0*(C1*np.sqrt(xCoords) + C2*xCoords + C3*xCoords**2 + C4*xCoords**3 + C5*xCoords**4)
            
    def computeArclengthInterpolation(self, xCoords, zCoords, numPoints):
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
        zPoints = self.generateNacaPoints(xPoints)
        return xPoints, zPoints

    def computeColocationPoints(self, xCoords, zCoords, chordLength):
        scaleX = COLOCATION_SCALING_FACTOR_X
        scaleZ = COLOCATION_SCALING_FACTOR_Z
        midPointShift = 0.5 * (1 - scaleX) * chordLength
        colocationXUpper = scaleX * xCoords + midPointShift
        colocationXLower = scaleX * xCoords[-1:0:-1] + midPointShift
        colocationZUpper = scaleZ * zCoords
        colocationZLower = -scaleZ * zCoords[-1:0:-1]
        return (np.concatenate((colocationXUpper, colocationXLower)),
                np.concatenate((colocationZUpper, colocationZLower)))
                
    def generateElementMatrices(self):
        self.numElements = (self.numPoints - 1) // (NUM_POINTS_IN_ELEMENT - 1)
        connectionMatrix = np.zeros((self.numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
        assemblyMatrix = np.zeros((self.numElements, NUM_POINTS_IN_ELEMENT), dtype=int)
    
#         Generate the connection matrix to make the trailing elements not have the same last point
#       Generate the assembly matrix to place the value of the trailing points in the same index
        halfElements = self.numElements // 2
        for i in range(halfElements):
            start = i * 2
            connectionMatrix[i] = np.arange(start, start + NUM_POINTS_IN_ELEMENT)
            connectionMatrix[i + halfElements] = np.arange(
                start + (self.numPoints // 2) + 1, start + (self.numPoints // 2) + NUM_POINTS_IN_ELEMENT + 1)
            assemblyMatrix[i] = np.arange(start, start + NUM_POINTS_IN_ELEMENT)
            assemblyMatrix[i + halfElements] = np.arange(
                start + self.numPoints // 2, start + self.numPoints // 2 + NUM_POINTS_IN_ELEMENT)
        connectionMatrix[-1][-1] = 0
        assemblyMatrix[-1][-1] = 0
        
        self.geometry.setElementMatrices(connectionMatrix, assemblyMatrix)

    
