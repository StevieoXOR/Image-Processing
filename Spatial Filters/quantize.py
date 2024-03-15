# Look at readImg.py to get a better grasp of image processing, numpy arrays, etc

import numpy as np
picURL = ".\\Lena.png"	# . = local directory (folder) that this file is in
bitPrecision_LenaImg = np.uint8

from PIL import Image
# Must have .convert('L') since srcImg is RGB instead of grayscale. L=Luminance=Intensity=Brightness
originalImgArr_2d = np.asarray( Image.open(picURL).convert('L'), dtype=bitPrecision_LenaImg )
origImgArr_1d = originalImgArr_2d.flatten()	# 2D-img -> 1D-img


pixelsPerImg = len(origImgArr_1d)
numIntensities = 256
numLevels = 8
intensitiesPerLevel = (numIntensities/numLevels)


def quantize():
	# Each pixel will be a certain intensity, specifically within [0,numIntensities)
	# Uniform Quantization is a uniform intensity mapping.
	# I.e., the intensity range is split into equal portions.
	# [0-7],[8-15],[16-23],[24-31], ..., [248-255]
	# =   0,     1,      2,      3, ...,         7
	#if   ((imgAry_1d[i])//numLevels) == 0:	newImgAry_1d[i] = 0*intensitiesPerLevel
	#elif ((imgAry_1d[i])//numLevels) == 1:	newImgAry_1d[i] = 1*intensitiesPerLevel
	#elif ((imgAry_1d[i])//numLevels) == 2:	newImgAry_1d[i] = 2*intensitiesPerLevel
	#elif ((imgAry_1d[i])//numLevels) == 3:	newImgAry_1d[i] = 3*intensitiesPerLevel
	#...
	
	# For each pixel, check for the quantizationLevel it fits into.
	# Once the quantLvl matches, copy that quantLvl into the corresponding pixel of the new image
	newImgAry_1d = np.zeros_like(origImgArr_1d)	# Create array with same length as srcImg
	for pixelIdx in range(pixelsPerImg):	# 0 to hella
		currPixelIntensity = origImgArr_1d[pixelIdx]
		for i in range(numLevels):		# 0 to 7
			if (currPixelIntensity//numLevels) == i:
				newImgAry_1d[pixelIdx] = int(i*intensitiesPerLevel)
				#print(i*intensitiesPerLevel)
#	for p in newImgAry_1d:
#		print(str(p)+", ", end="")

	return newImgAry_1d


def getMeanSqrError(img1_1d, img2_1d):
	cumSum = 0
	for pixelIdx in range(pixelsPerImg):
		# The below line is commented out because each pixel intensity was
		#   interpreted as a fixed-precision np# instead of any-size integer.
		#diff = (quantizedImg_1d[pixelIdx] - origImgArr_1d[pixelIdx])
		diff = int(img1_1d[pixelIdx]) - int(img2_1d[pixelIdx])
		cumSum += diff*diff
	mse = cumSum//pixelsPerImg
	return mse

quantizedImg_1d = quantize()
#print( "Difference between Original and Quantized:", getMeanSqrError(quantizedImg_1d,origImgArr_1d) )


def getShiftedAndScaledRandNum():
	# Generate random int between 0 and 8 - [0,8]
	randNum = np.random.randint(9)
	
	# Shift the range down by half to get [-4,4]
	randNum -= 4

	# Scale it up (or down) to match a uniform distribution (0to8 is not uniform out of the set of 0to256)
	randNum *= intensitiesPerLevel

	return randNum

def noisifyImg(imgArr_1d):
	noisyImgAry_1d = np.zeros_like(imgArr_1d)	# Create array with same length as srcImg
	for pixelIdx in range(pixelsPerImg):
		noise = getShiftedAndScaledRandNum()
		noisyImgAry_1d[pixelIdx] = imgArr_1d[pixelIdx] + noise
	return noisyImgAry_1d


noisyImgAry_1d = noisifyImg(origImgArr_1d)
#print( "Difference between Noisy and Original:", getMeanSqrError(noisyImgAry_1d,origImgArr_1d) )

#print( "Difference between Noisy and Quantized:", getMeanSqrError(noisyImgAry_1d,quantizedImg_1d) )





#      [1 2 1]
#  1/16[2 4 2]
#      [1 2 1]
# kernCoeff==kernel coefficient=(1/16)
# kern==kernel=matrixWithoutCoefficient
# bigMat==big matrix==image = unshown in this example
def matrixConvolve(kernCoeff, kern, bigMat):
	outputImgAry_2d = np.zeros_like(bigMat)	# Create array with same height and width as bigMatrix
	R = len(bigMat)		#R=numRows in bigMatrix
	C = len(bigMat[0])	#C=numCols
	kernR = len(kern)	#R=numRows in convolutionMatrix
	kernC = len(kern[0])	#C=numCols
	if (kernR != 3) or (kernC != 3): return "unimplemented kernel size"

	# Definitely doesn't work since kernel and kernCoeff aren't being multiplied by anything, but it shows you the general idea
	#for rowIdx in range(R-kernR):
	#	for colIdx in range(C-kernC):
	#		sum_3x3 = bigMat[rowIdx  ][colIdx : 3+colIdx]\
	#			+ bigMat[rowIdx+1][colIdx : 3+colIdx]\
	#			+ bigMat[rowIdx+2][colIdx : 3+colIdx]
	#		#bigMat[rowIdx+1][colIdx+1] = sum_3x3
	#		print(sum_3x3)
	
	# Specifically for a 3x3 convolution matrix ("convolution kernel")
	for r in range(R-kernR+1):	# for r in bigMatrixRows (accounting for kernel size)
	  for c in range(C-kernC+1):	# for c in bigMatrixCols (accounting for kernel size)
	    wsum_3x3 = bigMat[r  ][c] * kern[0][0]  +   bigMat[r  ][c+1] * kern[0][1]  +  bigMat[r  ][c+2] * kern[0][2]\
		     + bigMat[r+1][c] * kern[1][0]  +   bigMat[r+1][c+1] * kern[1][1]  +  bigMat[r+1][c+2] * kern[1][2]\
		     + bigMat[r+2][c] * kern[2][0]  +   bigMat[r+2][c+1] * kern[2][1]  +  bigMat[r+2][c+2] * kern[2][2]
	    print(wsum_3x3)
	    
	    # outputMatrix_middleElementOfFilter = convolutionResult
	    outputImgAry_2d[r+1][c+1] = (wsum_3x3 * kernCoeff)
	    print(outputImgAry_2d[r+1][c+1])
	    print(outputImgAry_2d)
	    print()

	# This is a stepping stone for future improvement
	#for r in range(R-kernR+1):	# for r in bigMatrixRows (accounting for kernel size)
	# for i in range(kernR):		# for i in kernelRows
	#  for c in range(C-kernC+1):
	#   for j in range(kernC):
	#    wsum_3x3 = bigMat[r  ][c] * kern[i  ][j]  +   bigMat[r  ][c+1] * kern[i  ][j+1]  +  bigMat[r  ][c+2] * kern[i  ][j+2]\
	#	      + bigMat[r+1][c] * kern[i+1][j]  +   bigMat[r+1][c+1] * kern[i+1][j+1]  +  bigMat[r+1][c+2] * kern[i+1][j+2]\
	#	      + bigMat[r+2][c] * kern[i+2][j]  +   bigMat[r+2][c+1] * kern[i+2][j+1]  +  bigMat[r+2][c+2] * kern[i+2][j+2]
	#    print(wsum_3x3)
	#    
	#    # outputMatrix_middleElementOfFilter = convolutionResult
	#    outputImgAry_2d[r+1][c+1] = (wsum_3x3 * kernCoeff)



kern=[[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]
bigMatrix = np.ones(shape=(5,5), dtype=np.uint8)
bigMatrix = originalImgArr_2d

#[[1*1 2*2 3*1],[4*2 5*4 6*2],[7*1 8*2 9*1]] => [[1 4 3],[8 20 12],[7 16 9]] => Sum=8+40+32=80 => 80/16=5
# So, center element should be 5. Since it is, that shows the program is working.
bigMatrix = [[1,2,3],[4,5,6],[7,8,9]]
matrixConvolve(1,kern,bigMatrix)


def getMSEofConvolutions():
	from scipy import signal
	numRows = len(originalImgArr_2d)
	numCols = len(originalImgArr_2d[0])
	
	# Convolve (Pointwise matrix multiply) the Quantized and Noisy matrices with the kernel given by the problem
	# Arrays must be matrices (2D) in order to be convolved using `scipy.signal`
	quant_2d = quantizedImg_1d.reshape(numRows,numCols)
	convQuant_imgArr_2d = signal.convolve2d(quant_2d, kern, boundary='symm', mode='same')
	
	noisy_2d = noisyImgAry_1d.reshape(numRows,numCols)
	convNoisy_imgArr_2d = signal.convolve2d(quant_2d, kern, boundary='symm', mode='same')
	
	
	# Convert arrays back to 1d arrays so I can use the MSE function again
	convQuant_imgArr_1d = convQuant_imgArr_2d.flatten()
	convNoisy_imgArr_1d = convNoisy_imgArr_2d.flatten()
	print( "Difference between originalImg and convolvedNoisy:",
		getMeanSqrError(origImgArr_1d, convNoisy_imgArr_1d) )
	print( "Difference between originalImg and convolvedQuantized :",
		getMeanSqrError(origImgArr_1d, convQuant_imgArr_1d) )
	print( "Difference between convolvedNoisy and convolvedQuantized:",
		getMeanSqrError(convQuant_imgArr_1d, convNoisy_imgArr_1d) )

#getMSEofConvolutions()