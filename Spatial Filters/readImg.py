import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


##### Global(ish) variables
# . = local directory (folder) that this file is in
#picURL = "C:\\Users\\someUser\\What A Beautiful Folder\\Image Processing\\Another Folder\\box.png"
picURL = ".\\box.png"
#picURL = ".\\Bean.jpg"
#picURL = ".\\Lena.png"
picURL = ".\\Test.png"
#picURL = ".\\Dog.jpg"
bitPrecision_boxImg = np.uint8		# 8 bits per pixel
saveModdedImg = True


# https://stackoverflow.com/questions/72459847/how-to-save-a-tkinter-canvas-as-an-image
# Unitized  = Made all values between 0 and 1, preserving the relationship (usually a ratio) between values
# Equalized = Transforming the relationship between values, keeping the structure relatively intact but still modifying the structure.
# A transformation function is NOT the resulting image. A transformation function is what's used to MAP a source image to a resulting image.
# Function = Either a Mathematical function (e.g. f(x)) or a Programming function (e.g. doThing()).

# Notes:
# To understand the CDF fully, take a statistics (or probability or stochastics) course.
#	(Graphs) and (numerical fully-written-out examples using arrays) are the key to understanding the CDF.
# When I mention "pixel-to-pixel ratio is/n't preserved", that's important
#	because maintaining roughly the same ratio as the original image leads to a
#	very similar-looking output image (compared to input image), not random junk

# Housekeeping:
# Once this file actually gets large, I will convert it into a class format and
#	organize the transformation functions into its own class, and put the
#	visualization functions into their own class.


# Non-openCV way
###########################################
#     IMAGE DRAWING HELPER FUNCTIONS      #
###########################################
# https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib  cxrodgers's answer
""" Way to set where the matplotlib window spawns on the screen"""
def moveFigureTo(fig, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = mpl.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        fig.canvas.manager.window.move(x, y)


def drawHistogram(hist, windowName="Histogram"):
	import tkinter as tk
	# draw = ImageDraw.Draw(img)		# Might use in the near future
	# display = ImageTk.PhotoImage(img)	# Might use in the near future

	canvWidth = 650
	canvHeight = 400
	windowMargin = 50
	y_bottom = canvHeight-windowMargin		# Start at almost the bottom of the window
	steps = numIntensities					# Number of x-values in histogram
	step_size = (canvWidth-2*windowMargin) // steps		# For visuals only
	
	# Max height of any individual column in the histogram should be as tall as the histogram, NOT TALLER.
	scalableHist = unitize1Darray(hist)
	vertScale = canvHeight-(2*windowMargin)
	# if useScaleMultiplier:	vertScale = canvHeight-(2*windowMargin)
	# else:			vertScale = 1		# Already scaled before this function was called, so don't scale again.

	root = tk.Tk()
	root.title(windowName)
	canv = tk.Canvas(root, bg="white", width=canvWidth, height=canvHeight)	# Set canvas's size, color, owner
	root.geometry(str(canvWidth)+"x"+str(canvHeight))			# Set window's size
	
	# Draw x-axis
	bottomHorizAxisLine = [(windowMargin,canvHeight-windowMargin),
		(windowMargin+(step_size*steps),canvHeight-windowMargin)]
	canv.create_line(bottomHorizAxisLine, fill="red",width=1)
	
	# Draw each column in the histogram, including axis tick marks
	for x in range(windowMargin, canvWidth, step_size):
		pixelIntensity = (x-windowMargin) // step_size
		if pixelIntensity < numIntensities:
			y_end = y_bottom - int((scalableHist[pixelIntensity])*vertScale)
			vertLine = [(x, y_bottom), (x, y_end)]
			canv.create_line(vertLine, fill="black",width=1)
#			print(pixelIntensity, vertLine)	##########
	
			minorTick = [(x, y_bottom+2), (x, y_bottom+5)]
			canv.create_line(minorTick, fill="red",width=1)

	label = tk.Label(root)
	label.pack()
	canv.pack()
	root.mainloop()
# End drawHistogram()



###########################################
#    IMAGE PROCESSING HELPER FUNCTIONS    #
###########################################
"""Purpose:	Convert sourceImage directly into resultantImage using the Gamma Transformation Function.

	Inputs: Flattened 1D-array of original image,
			Gamma Coefficient ( coeff>0 ),
			Gamma Exponent ( 0<=exponent<infinity),
			Number of intensities in resultant image (numIntensities>0, must be int).
	Output: Flattened 1D-array of the transformed image.

	Notes:	THIS FUNCTION DOESN'T USE A HISTOGRAM. IT USES THE RAW IMAGE AS A 1D ARRAY.
			Gamma Mapping Function  ==  dstIntensity = coefficient*(srcIntensity^exponent), where
			srcIntensity = intensity of a specific pixel (location-wise) in the source image     and
			dstIntensity = intensity of the same specific pixel (location-wise) in the resultant (destination) image
"""
def applyTransformationFunc_Gamma(np1Darray, coeff, exponent, numIntensities:int):
	# Use (not get) a transformation function. We're multiplying intensities.
	# We're remapping (converting) intensities using this function.
	if exponent<0:
		print("Gamma must be between 0 and +infinity, with 1 as a middle point of no-change")
		exit(1)
	elif numIntensities<1:
		print("numIntensities must be an integer greater than 0")
		exit(1)

	convertedImgArr_1d = np.zeros_like(np1Darray)
	for i in range( len(np1Darray) ):
		convertedImgArr_1d[i] = coeff*( (np1Darray[i])**exponent)
	
	# Normalize
	# Each element in the array gets divided by brightestIntensityInTransformedImage to get between 0 and 1
	# Pixel-to-pixel intensity ratios are preserved.
	brightestTransformedPixel = np.max(convertedImgArr_1d)	# Get biggest (intensity) value in transformedImg
	convertedImgArr_1d = convertedImgArr_1d / brightestTransformedPixel

	# Re-scale
	# Each element in the array gets multiplied by the maximum intensity value
	# Image is now not extremely dark (low intensity values for all pixels = dark)
	convertedImgArr_1d = (numIntensities-1) * convertedImgArr_1d

	# Convert potentially negative float intensities to positive int intensities (you can't have a non-integer nor negative brightness of a pixel)
	convertedAndDiscretizedImgArr_1d = np.zeros_like(np1Darray)	# New array required because of float-type vs int-type arrays
	for pixelIdx in range( len(convertedImgArr_1d) ):
		convertedAndDiscretizedImgArr_1d[pixelIdx] = abs(  int( round(convertedImgArr_1d[pixelIdx]) )  )
#	print("Gamma-transformed 1D Array of Image: "+str(convertedAndDiscretizedImgArr_1d))	############
	
	return convertedAndDiscretizedImgArr_1d	# NOT an equalizedHistogram, NOR a kind of histogram, but is the actual (modified) image itself


"""Purpose:	Get a Gamma transformation function (i.e., 1D-array for mapping).
			We're creating a mapping array (i.e., 1D-array) for converting srcImg intensities to resultantImg intensities.
			Using the result of this function, you can remap (convert) intensities.

	Inputs: Gamma Coefficient ( coeff>0 ),
			Gamma Exponent ( 0<=exponent<=128),
			Number of intensities in resultant image (numIntensities>0, must be int).
	Output: Mapping array, where the array's input (i) will be a source image's
	 			intensity value and each output (Arr[i]) will be a transformed image's intensity value.
			This (programming) function does NOT produce any kind of image!

	Notes:	THIS TRANSFORMATION FUNCTION DOESN'T USE (A HISTOGRAM) NOR (A RAW IMAGE AS A 1D ARRAY).
			Gamma Mapping Function  ==  dstIntensity = coefficient*(srcIntensity^exponent), where
				srcIntensity = intensity of a specific pixel (location-wise) in the source image     and
				dstIntensity = intensity of the same specific pixel (location-wise) in the resultant (destination) image
"""
def getTransformationFunc_Gamma(coeff, exponent, numIntensities):
	# Get a transformation function. We're not multiplying intensities.
	# We're creating a mapping array for converting srcImg intensities to resultantImg intensities.
	if exponent<0 or exponent>128:
		print("In function:  getTransformationFunc_Gamma()")
		print("* Gamma (the exponent) must be between 0 and +infinity, with 1 as a middle point of no-change to image.\n"\
			+"* Due to practical limitations of storing very large numbers in arrays, Gamma also\n"\
			+"*   cannot be larger than 128 when using numIntensities=256.\n"\
			+"* If you use a larger value for numIntensities, the transformation array will be\n"\
			+"*   longer, and therefore the value produced by coeff*(maxIntensity^Gamma) in the\n"\
			+"*   last element of the transformation array will be larger, causing overflow sooner than expected.")
		exit(1)
	elif numIntensities<1:
		print("numIntensities must be an integer greater than 0")
		exit(1)

	# array of (length=numIntensities) 0s
	# transformationArr_1d: I had overflow issues with large exponents' results
	#	being converted to 64-bit floats (np.double) and even unsigned 64-bit ints
	#	(np.uint64) when using 256 possible intensities, so I'm not sure what
	#	implicit internal datatype is used in the transformationArr_1d array.
	transformationArr_1d			= np.zeros(numIntensities)
	discretizedTransformationArr_1d	= np.zeros(numIntensities, dtype=np.uint8)	# 8-bit positive int

	# Pixel-to-pixel intensity ratios are NOT preserved in this step since this is a *transformation* function.
	for i in range( len(transformationArr_1d) ):
		transformationArr_1d[i] = coeff*( i**exponent)
	# transformationArr_1d  =  [0  c*1^exp  c*2^exp  c*3^exp  c*4^exp  ...  c*(numIntensities-1)^exp].
	
	# Normalize
	# Each element in the array gets divided by brightestIntensityInTransformedImage to get between 0 and 1
	# Pixel-to-pixel intensity ratios are preserved in this step.
	brightestTransformedPixel = transformationArr_1d[-1]	# Get biggest (intensity) value in transformationArr_1d
	transformationArr_1d = transformationArr_1d / brightestTransformedPixel

	# Re-scale
	# Each element in the array gets multiplied by the maximum intensity value,
	#	then each intensity is forced to be a positive (or 0) value via abs() (because negative coefficients are allowed)
	#	then each intensity is forced to be an integer so the post-transformation-mapping can be read as an image
	# Image is now not extremely dark (low intensity values for all pixels = dark)
	#
	# Pixel-to-pixel intensity ratios are preserved in this step.
	# Operation that modifies entire array:  resultArray = coefficient*entireArray
	#	E.g.  resultArray = [c*arr[0]  c*arr[1]  c*arr[2]  ...  c*arr[-1]]
	transformationArr_1d = (numIntensities-1) * transformationArr_1d
	#
	# Pixel-to-pixel intensity ratios are *mostly* preserved in this step.
	#	I'm unsure if banker's rounding is used, which would help prevent artifacts
	#	(non-random patterns across the img that produce aliasing (waves or streaks)).
	for i in range( len(transformationArr_1d) ):
		discretizedTransformationArr_1d[i] = int(round(  abs( transformationArr_1d[i] ) ))

	return discretizedTransformationArr_1d	# NOT an equalizedHistogram, NOR a kind of histogram. A mapping array.


"""Purpose:	Create a CDF transformation 'function' (i.e., 1D-array).
			We're creating a mapping array (i.e., 1D-array) for converting srcImg intensities to resultantImg intensities.
			Using the result of this function, you can remap (convert) intensities.

	Inputs: np1Darray_hist = Histogram of flattened array of original-image,
			numIntensities = Number of representable intensities in final (transformed) image.
	Output: CDF of histogram of original image (not a usable output image!),
			*	I.e., Mapping array, where the array's input (i) will be a source image's
	 				intensity value and each output (Arr[i]) will be a transformed image's intensity value.
			*	I.e., a transformation function that can be used to convert an input image into a transformed image.
			This (programming) function does NOT produce any kind of image!

	Notes:	THIS FUNCTION USES A 1D-HISTOGRAM AS INPUT; IT DOES NOT USE THE RAW IMAGE AS A 1D ARRAY.

"""
def getTransformationFunc_CDF(np1Darray_hist, numIntensities):
	# I chose to find the CDF transformation function.
	# https://www.youtube.com/watch?v=uqeOrtAzSyU

	# np1Darray_hist is a histogram, but each of the (probably 256) intensities has a pixelCount.
	#	Sum of multiple intensities: Units are "#pixels"
	# CDF = Cumulative Distribution Function = Cumulative Sum (cumsum) = Transformation Function:
	# Example of CDF:   srcImg=[1,3,7,2]  ->  cdfTransformationFunction=[1,1+3,1+3+7,1+3+7+2]=[1,4,11,13]
	#											(resultantImg not included in this example)
	cdf_arr = np.cumsum(np1Darray_hist)		# cdf_arr == Transformation Function == Mapping Function
	# cdf_arr = [0  a[0]  a[0]+a[1]  a[0]+a[1]+a[2]  a[0]+a[1]+a[2]+a[3]  ...  a[0]+a[1]+a[2]+a[3]+...+a[-1]]
	#	where a == np1Darray_hist
	# Pixel-to-pixel intensity ratios are NOT preserved in this step since this is a *transformation* function.


	# Last element (a[-1]) of the CDF has the sum of every value in the PDF (PDF=histogram in this case)
	# PDF = Probability Density Function
	numPixelsInImg = cdf_arr[-1]
	
	# Normalize
	# Each element in the array gets divided by numPixelsInImg to get between 0 and 1
	# Pixel-to-pixel intensity ratios are preserved in this step.
	cdf_arr = cdf_arr / numPixelsInImg

	# Re-scale
	# Each element in the array gets multiplied by the maximum intensity value
	#	(so that the full scale of the histogram (from 0 to numIntensities-1) can be reached)
	# Pixel-to-pixel intensity ratios are preserved in this step.
	cdf_arr = (numIntensities-1) * cdf_arr
	
	return cdf_arr	# NOT an equalizedHistogram, NOR a kind of histogram. A mapping array.


"""Purpose: Uses the Transformation Function (like CDF(histogram-based) or Gamma(image-independent))
				to convert srcImg intensities to resultantImg intensities.

	Inputs: Flattened 1D-array of original-image,
			1D-array Transform (Mapping) 'function' that may or may not be based on original image
			* like CDF(histogram-based) or Gamma(image-independent),
			Transformation type:  None, 'c' or 'cdf', 'g' or 'gam' or 'gamma'.
	Outputs: Flattened 1D-array of transformed image.
"""
def equalizeImgUsingTransformFunction(srcImgArr_1d, transformArr_1d, trnsfmType=""):
	# https://stackoverflow.com/questions/46945687/histogram-equalization-of-image-using-lookup-table
	# https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23
	# Vaguely-typed way to do what's below:
	#	newImg = cdf[srcImg], where the CDF is a Transformation (mapping) function
	#	newImg = gamma[srcImg], where the Gamma is a Transformation (mapping) function
	usePrint = False		# Meant for debugging	##############################

	# Create new array (of all 0s) that has same size as the image
	newImg_1dArr = np.zeros_like(srcImgArr_1d)

	for pixelIndex in range( len(srcImgArr_1d) ):
		currPixelIntensity = srcImgArr_1d[pixelIndex]
		newImg_1dArr[pixelIndex] = transformArr_1d[currPixelIntensity]	# Lookup-table.

		t = trnsfmType.lower()	# Make the string lowercase
		if usePrint:
			if t=="":
				print("newImg[i]="+str(newImg_1dArr[pixelIndex])+"  <==  histogramTransform[originalImg]="+str(currPixelIntensity))		#######
			if t=="c" or t=="cdf":
				print("newImg[i]="+str(newImg_1dArr[pixelIndex])+"  <==  histogramTransformCDF[originalImg]="+str(currPixelIntensity))	#######
			if t=="g" or t=="gam" or t=="gamma":
				print("newImg[i]="+str(newImg_1dArr[pixelIndex])+"  <==  transformGamma[originalImg]="+str(currPixelIntensity))			#######
	return newImg_1dArr


def getHistogram(imgArr_1d, numIntensities):
	# Create histogram array with length=numIntensities
	# Histogram: Each element on the x-axis gets its own counter.
	#	An image with 234 pixels (out of any bigger number of total pixels)
	#	that each have intensity==1 means histArray[1]==234.
	histogram = np.zeros(numIntensities,dtype=np.uint32)
	for pixelIntensity in imgArr_1d:
		histogram[int(pixelIntensity)] += 1
#		print(pixelIntensity)	###########
	return histogram

def unitize1Darray(arr_1d):
	unitizedArr_1d = np.zeros_like(arr_1d, dtype=np.double)
	# Find largest num of pixels with a single intensity.
	# This will be the tallest value in the graph, so use it to unitize (make between 0 and 1) the entire graph.
	#maxCountForSingleIntensity = 0
	#for pixelIntensity in range(len(histogram)):
	#	if arr_1d[pixelIntensity] > maxCountForSingleIntensity:
	#		maxCountForPixelsWithSingleIntensity = arr_1d[pixelIntensity]
	maxCountForPixelsWithSingleIntensity = arr_1d.max()	# Same as above lines that are commented out

	for i in range(len(arr_1d)):
		unitizedArr_1d[i] = (arr_1d[i])/maxCountForPixelsWithSingleIntensity
#	print(unitizedArr_1d)	#########
	
	return unitizedArr_1d

# function(thingThatMustBePassedIn, thingWithDefaultValueSoItDoesNotHaveToBePassedIn=DefaultValue)
def myMethod(numIntensities, drawHist, useCDFtransform=False, useGammaTransform=False, gammaArgs=None):
	# PIL = Python Imaging Library, which has nowadays been renamed to Pillow but uses the same import for compatibility
	from PIL import Image, ImageDraw, ImageTk
	
	# L = Luminance = Intensity = Brightness of a pixel ==> Monochrome = Grayscale (Black/White/InBetween)
	img = Image.open(picURL).convert('L')
	
	# Get pixel intensity values of image, put them into an array
	# Default array precision is int32, so change to whatever precision the image uses.
#	arr = np.array(img.getdata(), dtype=bitPrecision_boxImg)
	# OR
	imgArr_2d = np.asarray( img, dtype=bitPrecision_boxImg )
	imgArr_1d = imgArr_2d.flatten()	# 2D-img -> 1D-img  (how it works: Each row gets concatenated (appended) to previous row)

#	numRows = len(imgArr_2d)	# len([[rowStart,...,rowEnd],[rowStart,...,rowEnd],...])
#	numCols = len(imgArr_2d[0])	# len([rowStart,...,rowEnd])
#	print( numRows, numCols )	##########

	# Create 1D-histogram array of source(original)(unmodified) image with length=numIntensities
	histogram = getHistogram(imgArr_1d, numIntensities)
	
	
	######
	unitizedHistogram = unitize1Darray(histogram)
	# Find largest num of pixels with a single intensity.
	# This will be the tallest value in the graph, so use it to unitize (make between 0 and 1) the entire graph.
#	print(unitizedHistogram)	#########
	######
	######
	# Initial values
	equalizedImg_1d = None
	useImgTransformation = False
	
	if useCDFtransform and useGammaTransform:
		print("You have chosen to use histogram equalization, but you have chosen multiple methods.")
		print("Pick only one Transform to use.")
		exit(1)
	elif useCDFtransform:
		print("You have chosen to use histogram equalization via CDF-histogram transformation")
		useImgTransformation = True
		CDFtransformArr_1d = getTransformationFunc_CDF(histogram,numIntensities)
		equalizedImg_1d = equalizeImgUsingTransformFunction(imgArr_1d, CDFtransformArr_1d)
	elif useGammaTransform:
		if len(gammaArgs)!=2:
			print("Using the Gamma transform. Expected gammaArgs=[transformCoefficient, transformExponent]")
			exit(1)
		print("You have chosen to use histogram equalization via Gamma transformation"+\
			f" with   adjustedIntensity = {gammaArgs[0]}(originalIntensity)^{gammaArgs[1]}")
		useImgTransformation = True
		# The below commented-out method has the same result as performing the (not-commented-out) mapping transformation below
		# GammaTransformedArr_1d = applyTransformationFunc_Gamma(imgArr_1d, gammaArgs[0], gammaArgs[1], numIntensities)
		# equalizedImg_1d = GammaTransformedArr_1d
		GammaTransformationArr_1d = getTransformationFunc_Gamma(gammaArgs[0], gammaArgs[1], numIntensities)
		equalizedImg_1d = equalizeImgUsingTransformFunction(imgArr_1d, GammaTransformationArr_1d)
		# BELOW: THE TRANSFORMATION HAS ALREADY BEEN APPLIED BY USING THE GAMMA FUNCTION. THIS IS COMPLETELY WRONG.
		# equalizedImg_1d = equalizeImgUsingTransformFunction(imgArr_1d, GammaTransformedArr_1d)
	else:
		print("You have chosen to NOT use any form of histogram equalization nor transformation.")

	if useImgTransformation:
		# print("equalizedImg_1d: "+str(equalizedImg_1d))##########
		# put array back into original 2D shape since we flattened it
		equalizedImg_2d = np.reshape(equalizedImg_1d, imgArr_2d.shape)
	######

	
	# Display regular picture's histogram, maybe also equalized-histogram. Uses Tkinter.
	if drawHist:
		drawHistogram(hist=unitizedHistogram, windowName="Unmodified Histogram of Image")
		if useImgTransformation:
			drawHistogram(hist=getHistogram(equalizedImg_1d, numIntensities),
				windowName="Transformed Histogram of Image")
	
	if saveModdedImg:
		# This is a 2d array. Do NOT use a 1d array, which will cause each
		#	pixel to be treated as its own row, meaning the image will be a
		#	single pixel wide (a single column).
		im = Image.fromarray(equalizedImg_2d)
		# print(im.size)	# Print numPixels for each dimension.

		modType = ""
		if useCDFtransform:
			modType = "CDF"
		elif useGammaTransform:
			modType = "Gamma"
		
		fileSuffix = ".jpg"	# Cannot be JPG if (>65500 pixels PER DIMENSION)
		# Discard the ".\\" of the filename by "picURL[2:]"  ("\\" is a single character long").
		# Discard the three-letter file suffix and the dot by "picURL[:-4]"
		filename_savedModdedImg = picURL[2:-4] + " " + modType + "Equalized" + fileSuffix
		im.save(filename_savedModdedImg, quality=100)	# 100% compression quality (if lossy method is used)


	# Display regular picture, maybe also equalized-histogram picture. Uses matplotlib.
	fig = plt.figure()
	moveFigureTo(fig, 10, 10)	# 10,10 is the screen position where the top-left of the whole window will be placed
	if useImgTransformation:
		# set up side-by-side image display
		fig.set_figheight(8)
		fig.set_figwidth(18)

		# Display original (unequalized) image
		fig.add_subplot(1,2,1)	# rows,cols,index
		plt.imshow(imgArr_2d, cmap='gray')

		# Display equalized image
		fig.add_subplot(1,2,2)
		plt.imshow(equalizedImg_2d, cmap='gray')
	else:
		fig.set_figheight(15)
		fig.set_figwidth(15)
		
		# Display original (unequalized) image
		plt.imshow(imgArr_2d, cmap='gray')
		
	plt.show()	# Opens the window
# endFunction



def cvMethod(numIntensities, drawHist, useHistogramEqualization):
	import cv2

	im = cv2.imread(picURL)
	src = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	dst = cv2.equalizeHist(src)

	if drawHist:
		# calculate mean value array from RGB channels (convert to grayscale) and flatten to 1D array
		#src_1d = src.mean(axis=2).flatten()
		#dst_1d = dst.mean(axis=2).flatten()

		# flatten 2D image array to 1D array
		src_1d = src.flatten()
		dst_1d = dst.flatten()
		if useHistogramEqualization:
			# plot histogram with numIntensities=256 bins
			b, bins, patches = plt.hist([src_1d, dst_1d], numIntensities, label=['unequalized', 'equalized'])
		else:
			# plot histogram with numIntensities=256 bins
			b, bins, patches = plt.hist(src_1d, numIntensities, label='unequalized')
		plt.legend(loc='upper right')
		plt.xlim([0,255])
		plt.show()


	winWidth  = 800
	winHeight = 400
	title1 = "Unequalized image"
	title2 = "Equalized Image"

	# Naming a window so that I can resize it so it's visible, and I can drag it around and read its title
	cv2.namedWindow(title1, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(title1, winWidth, winHeight)
	cv2.imshow(title1, src)
	if useHistogramEqualization:
		cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(title2, winWidth, winHeight)
		cv2.imshow(title2, dst)
	cv2.waitKey(0)		# Press any key
	cv2.destroyAllWindows()


	


if __name__ == "__main__":
	# From the 8-bit precision of the image, each pixel can have one of 2^8=256 different intensities, from 0 to 255
	numIntensities = 256

	# Every one of these function calls is perfectly okay
	#myMethod(numIntensities, drawHist=True, useCDFtransform=True)
	# myMethod(numIntensities, drawHist=True, useCDFtransform=True,  useGammaTransform=False)
	# myMethod(numIntensities, drawHist=True, useCDFtransform=True,  useGammaTransform=False, gammaArgs=[5, 3])	
	# myMethod(numIntensities, drawHist=True, useCDFtransform=False, useGammaTransform=True, gammaArgs=[5, 3])
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, 128])
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, 25])
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, 1.5])	# Increase darkness
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, 1])		# No change
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, .75])	# Increase brightness
	myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, .5])	# Works really well on Test.something img
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, .25])
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, .1])
	# myMethod(numIntensities, drawHist=True, useGammaTransform=True, gammaArgs=[1, .01])

	# cvMethod(numIntensities, drawHist=True, useHistogramEqualization=True)