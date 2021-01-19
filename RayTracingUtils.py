import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def sign(x):
    return torch.sign(x)


class Display:
    def __init__(self, observerPositions, height=600, width=800, diffusionPower=[40, 0], halfPhysSize=[200, 150],
                 viewerDistance=400,
                 projectorResolution=[800, 600], primaryResolution=3):
        self.DiffusionPower = diffusionPower
        self.HalfPhysSize = halfPhysSize
        self.ViewerDistance = viewerDistance
        self.ProjectorResolution = projectorResolution
        self.observerPositions = observerPositions
        self.ProjectorPosition = self.observerPositions[0, :].reshape(1, 1, 1, 3).detach()

        self.ori = torch.zeros(primaryResolution * primaryResolution, height, width, 3,
                               device=device)
        self.direc = torch.zeros(primaryResolution * primaryResolution, height, width, 3,
                                 device=device)

        # ---
        # these values are for non-pinhole rendering, not really needed
        # Comment to save a bit of memory?
        # self.d = torch.zeros(primaryResolution * primaryResolution, height, width, 3,
        #                      device=device)
        # self.d[:, :, :, 0] = 1
        # self.outLambda = torch.zeros_like(self.direc[:, :, :, 0],  device=device)
        # self.z0 = self.ViewerDistance * torch.ones_like(self.outLambda,  device=device)
        # ----
        self.MinX = -self.HalfPhysSize[0]
        self.MinY = -self.HalfPhysSize[1]
        self.MaxX = self.HalfPhysSize[0]
        self.MaxY = self.HalfPhysSize[1]
        self.ImagePlaneDepth = self.ViewerDistance

        self.OriginX = observerPositions[0, 0]
        self.OriginY = observerPositions[0, 1]
        self.updateFlag = False
        self.posX = 0
        self.posY = 0
        self.posZ = 0

    def updateValues(self, viewNr):
        self.OriginX = self.observerPositions[viewNr, 0]
        self.OriginY = self.observerPositions[viewNr, 1]
        self.ProjectorPosition = self.observerPositions[viewNr, :].reshape(1, 1, 1, 3).detach()
        self.updateFlag = True

    def updateRaster(self, raster):
        lambdaX = raster[:, :, :, 0] / self.ProjectorResolution[0]
        lambdaY = 1 - raster[:, :, :, 1] / self.ProjectorResolution[1]
        self.posX = self.MinX + lambdaX * (self.MaxX - self.MinX)
        self.posY = self.MinY + lambdaY * (self.MaxY - self.MinY)
        self.posZ = self.ImagePlaneDepth

    def GenerateRay(self, raster):
        return 0
        lambdaX = -1.0 + 2.0 * raster[:, :, :, 0] / self.ProjectorResolution[0]

        lambdaY = 1.0 - 2.0 * raster[:, :, :, 1] / self.ProjectorResolution[1]

        # screen location
        x0 = lambdaX * self.HalfPhysSize[0]

        y0 = lambdaY * self.HalfPhysSize[1]

        z0 = self.z0.clone().detach()

        xyz = torch.stack((x0, y0, z0), -1)

        # viewer location

        locOri = self.ProjectorPosition - xyz

        locLineOri = -1 * xyz

        X = self.FindMaxOnLine(locOri, locLineOri)

        # this must be then pixel locaion
        self.ori[:, :, :, 0] = X
        self.ori[:, :, :, 1] = 0
        self.ori[:, :, :, 2] = 0
        self.direc[:, :, :, 0] = x0 - X
        self.direc[:, :, :, 1] = y0
        self.direc[:, :, :, 2] = z0

        return self.ori, self.direc

    def Diffusion(self, dirToProj, dirToEye):
        dA = dirToProj

        dB = dirToEye

        signA = sign(dA[:, :, :, 2])

        signB = sign(dB[:, :, :, 2])

        rhoA = signA * dA[:, :, :, 0] / torch.sqrt(
            dA[:, :, :, 1] * dA[:, :, :, 1] + dA[:, :, :, 2] * dA[:, :, :, 2] + 1e-12)

        rhoB = signB * dB[:, :, :, 0] / torch.sqrt(
            dB[:, :, :, 1] * dB[:, :, :, 1] + dB[:, :, :, 2] * dB[:, :, :, 2] + 1e-12)
        diffRho = rhoA - rhoB

        expArg = -self.DiffusionPower[0] * diffRho * diffRho

        if self.DiffusionPower[1] > 0:
            etaA = dA[:, :, :, 1] / (dA[:, :, :, 2] + 1e-12)

            etaB = dB[:, :, :, 1] / (dB[:, :, :, 2] + 1e-12)

            diffEta = etaA - etaB
            # diffRho = np.arctan(diffRho)
            # diffEta = np.arctan(diffEta)
            expArg -= self.DiffusionPower[1] * diffEta * diffEta

        return torch.exp(expArg).float()

    def FindMaxOnLine(self, rayOri, lineOri):
        signRay = sign(rayOri[:, :, :, 2])

        signLine = sign(lineOri[:, :, :, 2])

        rho = signRay * rayOri[:, :, :, 0] / torch.sqrt(
            rayOri[:, :, :, 1] * rayOri[:, :, :, 1] + rayOri[:, :, :, 2] * rayOri[:, :, :, 2])

        e = lineOri

        # use case for d != [1,0,0]
        # a = d[:, :, 0] * d[:, :, 0] - rho * rho * (d[:, :, 1] * d[:, :, 1] + d[:, :, 2] * d[:, :, 2])
        #
        # b = d[:, :, 0] * e[:, :, 0] - rho * rho * (d[:, :, 1] * e[:, :, 1] + d[:, :, 2] * e[:, :, 2])
        #
        # c = e[:, :, 0] * e[:, :, 0] - rho * rho * (e[:, :, 1] * e[:, :, 1] + e[:, :, 2] * e[:, :, 2])
        #
        # D = b * b - a * c
        #
        # # sometimes D can be less than zero..
        # DD = D<0
        # # D[D<0] = 0
        # sqrtD = torch.sqrt(D)
        # lambdaA = (-b + sqrtD) / a
        # lambdaB = (-b - sqrtD) / a
        lD = rho * torch.sqrt(e[:, :, :, 1] * e[:, :, :, 1] + e[:, :, :, 2] * e[:, :, :, 2])
        lambdaA = -e[:, :, :, 0] + lD
        lambdaB = -e[:, :, :, 0] - lD

        lambdaA = lambdaA.unsqueeze(-1)
        lambdaB = lambdaB.unsqueeze(-1)
        posA = e + lambdaA * self.d

        posB = e + lambdaB * self.d

        signA = sign(posA[:, :, :, 2])

        signB = sign(posB[:, :, :, 2])
        firstCondition = signA == signLine
        secondCondition = signB != signLine
        thirdCondition = signA != signLine
        fourthCondition = signB == signLine
        lambdaA = torch.squeeze(lambdaA, -1)
        lambdaB = torch.squeeze(lambdaB, -1)
        outLambda = self.outLambda.clone()
        outLambda[firstCondition * secondCondition] = lambdaA[firstCondition * secondCondition].float()
        outLambda[thirdCondition * fourthCondition] = lambdaB[thirdCondition * fourthCondition].float()

        stillZeros = outLambda == 0
        rhoA = signA * posA[:, :, :, 0] / torch.sqrt(
            posA[:, :, :, 1] * posA[:, :, :, 1] + posA[:, :, :, 2] * posA[:, :, :, 2])
        rhoB = signB * posB[:, :, :, 0] / torch.sqrt(
            posB[:, :, :, 1] * posB[:, :, :, 1] + posB[:, :, :, 2] * posB[:, :, :, 2])
        fifthCondition = torch.abs(rhoA - rho) < torch.abs(rhoB - rho)
        sixthCondition = ~fifthCondition
        outLambda[fifthCondition * stillZeros] = lambdaA[fifthCondition * stillZeros].float()
        outLambda[sixthCondition * stillZeros] = lambdaB[sixthCondition * stillZeros].float()

        return outLambda

    def GenerateRayPinHole(self):
        if not self.updateFlag:
            raise RuntimeError("Run display.updateValues first")

        self.ori[:, :, :, 0] = self.OriginX
        self.ori[:, :, :, 1] = self.OriginY
        self.ori[:, :, :, 2] = 0

        self.direc[:, :, :, 0] = self.posX - self.OriginX
        self.direc[:, :, :, 1] = self.posY - self.OriginY
        self.direc[:, :, :, 2] = self.posZ
        return self.ori, self.direc


class Sampler:
    def __init__(self, height=600, width=800, primaryResolution=3):
        self.secondaryRes = 1
        self.primaryResolution = primaryResolution
        with torch.no_grad():
            xs = torch.arange(0, width, device=device, dtype=torch.float32)
            ys = torch.arange(0, height, device=device, dtype=torch.float32)

            yv, xv = torch.meshgrid(ys, xs)

            self.raster = torch.stack((xv, yv), dim=2).unsqueeze(0).repeat(
                self.primaryResolution * self.primaryResolution, 1, 1, 1)

            for index in range(0, self.primaryResolution * self.primaryResolution):
                indY1 = index % self.primaryResolution

                indX1 = index // self.primaryResolution % self.primaryResolution

                dx1 = (0.5 + indX1) / self.primaryResolution

                dy1 = (0.5 + indY1) / self.primaryResolution
                self.raster[index, :, :, 0] += dx1
                self.raster[index, :, :, 1] += dy1
            self.raster[:, :, :, 0] = torch.clamp(self.raster[:, :, :, 0], 0, width)
            self.raster[:, :, :, 1] = torch.clamp(self.raster[:, :, :, 1], 0, height)

    def CurrentSample(self):
        return self.raster.detach()


class Renderer:
    def __init__(self, m_DisplayModel, sampler, projectorPositions, projectorImages, height=600, width=800,
                 primaryResolution=3):

        self.viewerDistance = m_DisplayModel.ViewerDistance
        self.halfSizeX = m_DisplayModel.HalfPhysSize[0]
        self.halfSizeY = m_DisplayModel.HalfPhysSize[1]
        self.sampler = sampler
        self.m_DisplayModel = m_DisplayModel
        self.projectorImages = projectorImages
        self.projectorPositions = projectorPositions
        self.primaryResolution = primaryResolution
        self.height = height
        self.width = width
        self.m_DisplayModel.updateRaster(self.sampler.CurrentSample())

    def updateProjectorImages(self, projectorImages, projectorPositions=None):
        self.projectorImages = projectorImages.permute(1, 2, 0)
        if projectorPositions is not None:
            self.projectorPositions = projectorPositions

    def projectPixels(self, xyz, ori):
        weightSum = torch.zeros((1, self.height, self.width), device=device).unsqueeze(-1)
        colorSum = torch.zeros((self.primaryResolution * self.primaryResolution, self.height, self.width, 3),
                               device=device)
        # xProj = torch.arange(0, 800).unsqueeze(0).unsqueeze(0).repeat(1, 600, 1).to(device)
        # yProj = torch.arange(0, 600).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 800).to(device)
        for projInd in range(self.projectorImages.size(2) // 3):
            projPos = self.projectorPositions[projInd]
            dirToProj = projPos - xyz
            dirToEye = ori - xyz
            weight = self.m_DisplayModel.Diffusion(dirToProj=dirToProj, dirToEye=dirToEye)
            # print(self.projectorImages[:,:,:].size())
            projColor = self.projectorImages[:, :, projInd * 3:(projInd + 1) * 3]
            # projColor = self.projectorImages[yProj, xProj, projInd * 3:(projInd + 1) * 3]

            weight = weight.unsqueeze(-1)
            weight[torch.isnan(weight)] = 0
            weight[weight < 0.00001] = 0
            weight[weight > 1e12] = 0
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # f = r - a  # free inside reserved
            # print(f)
            colorSum = colorSum + weight * projColor
            weightSum = weightSum + weight
        tmp = colorSum / (weightSum + 1e-12)
        tmp[(weightSum < 0.00001).expand(-1, -1, -1, 3)] = 0
        return tmp.float()

    def render(self, viewNr):

        self.m_DisplayModel.updateValues(viewNr)
        ori, direc = self.m_DisplayModel.GenerateRayPinHole()

        z0 = self.viewerDistance

        x0 = ori[:, :, :, 0] + (z0 - ori[:, :, :, 2]) * direc[:, :, :, 0] / direc[:, :, :, 2]

        y0 = ori[:, :, :, 1] + (z0 - ori[:, :, :, 2]) * direc[:, :, :, 1] / direc[:, :, :, 2]

        z0 = z0 * torch.ones_like(x0, device=device)
        xyz = torch.stack((x0, y0, z0), -1)

        return torch.mean(self.projectPixels(xyz=xyz, ori=ori), dim=0)
