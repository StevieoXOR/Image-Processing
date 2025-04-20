import easyocr as eo
import time

reader = eo.Reader(['en'])	# English



"""
Each BBox (bounding box) is tuple (xyTopLeft, xyTopRight, xyBotLeft, xyBotRight).
Each img can have multiple BBoxes.
Each BBox has an associated text prediction, as well as a probability of that text prediction.

img_paths: ["filename1.png", "filename2.JPG", ...]
results:[
[(img1_bbox_list_list, img1_str, img1_prob), (img2_bbox_list_list, img2_str, img2_prob), ...]
			[
				[bbox1_img1, text1_img1, prob1_img1], [bbox2_img1, text2_img1, prob2_img1]
			],
			[
				[bbox1_img2, text1_img2, prob1_img2]
			],
			[
				[bbox1_img3, text1_img3, prob1_img3], [bbox2_img3, text2_img3, prob2_img3], [bbox3_img3, text3_img3, prob3_img3]
			],
			...
		]
"""
"""
Supports visualizing results of multiple input images.
Uses Matplotlib to display resulting images
"""
def visualizeResults_Matplotlib(img_paths, results):
	import cv2
	import matplotlib.pyplot as plt
	import os

	# from pprint import pprint
	# pprint(results)
	# bounding_boxes_list_list = []
	# # bounding_boxes_list_list = [
	# # 	[(50, 30, 100, 100), (70,70,90,100)],  # boxes for img1
	# # 	[(60, 40, 100, 120)],                  # boxes for img2
	# # 	[(20, 10, 150, 90)],                   # boxes for img3
	# # ]
	# predicted_text_list_list = []
	# # predicted_text_list_list = [['This is', 'img1'], ['That is img2'], ['Malaysian pines img3']]
	# predicted_probs_list_list = []
	# # predicted_probs_list_list = [[]]

	# # Turn `results` into individual lists
	# for result in results:	# For each img in results
	# 	bbox_list = []
	# 	predicted_text_list = []
	# 	for (bbox, text, probabilityOfTextPrediction) in result:	# For each bounding box in a single img
	# 		bbox_list += [bbox]
	# 		predicted_text_list += [text]
	# 	bounding_boxes_list_list += [bbox_list]
	# 	predicted_text_list_list += [predicted_text_list]

	# if len(img_paths) != len(predicted_text_list_list):
	# 	raise ValueError("Each image must have a corresponding list of annotations (even if there is no predicted text or there's only one predicted token).")
	# if len(img_paths) != len(bounding_boxes_list_list):
	# 	raise ValueError("Each image must have a corresponding list of bounding boxes (even if there are 0 or 1 bounding boxes).")

	# Grid layout
	num_imgs = len(img_paths)
	cols = 2
	rows = (num_imgs + cols - 1) // cols

	# constrained_layout for preventing img and label overlap
	fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
	axs = axs.flatten()

	# For each img
	for idx, (img_path, result) in enumerate(zip(img_paths, results)):
		if not os.path.exists(img_path):
			raise FileNotFoundError(f"Image path does not exist: {img_path}")
		
		# Load image in OpenCV (BGR instead of RGB)
		img = cv2.imread(img_path)
		if img is None:
			raise ValueError(f"Could not load image: {img_path}")


		# print("result:",result)	# [(img1_bbox_list_list, img1_str, img1_prob), (img2_bbox_list_list, img2_str, img2_prob), ...]
		# boxesFor1Img, labelsFor1Img, probsFor1Img = result
		# ^^^ ValueError: too many values to unpack (expected 3). Causes entire img info to try to be put into boxesFor1Img,
		#             entire img info to be put into labelsFor1Img, same for probsFor1Img, and additional imgs are unexpected
		boxesFor1Img, labelsFor1Img, probsFor1Img = zip(*result)

		# Add probabilities to labels
		probsTrunc = []
		for prob in probsFor1Img:
			if prob>0.99:
				probsTrunc += ["prob=100%"]
			else:
				# Truncate all values past 2 decimal pts, then truncate the "0." at the start
				probsTrunc += ["prob="+f"{prob:0.2f}"[2:]+"%"]
		labelsWithProbsFor1Img = list(zip(labelsFor1Img, probsTrunc))
		str_labelsWithProbsFor1Img = str(labelsWithProbsFor1Img)[1:-1]	# Cut off outer "[]"

		# Word wrap:  Every 4 commas (2 phrases), add a newline to get word wrapping in labels so labels don't clip off the window
		cma_cnt = 0
		for i, c in enumerate(str_labelsWithProbsFor1Img):
			if c==',':
				cma_cnt+=1
				if cma_cnt%4 == 0:
					str_labelsWithProbsFor1Img = str_labelsWithProbsFor1Img[0:i+1]+"\n"+str_labelsWithProbsFor1Img[i+1:]

		# Draw bounding boxes with a black border
		BLACK, GREEN = (0, 0, 0), (0, 255, 0)
		for (xypair1, xypair2, xypair3, xypair4) in boxesFor1Img:
			topLeft, botRight = (int(xypair1[0]), int(xypair1[1])),  (int(xypair3[0]), int(xypair3[1])) # Create 2 tuples
			cv2.rectangle(img, topLeft, botRight, color=BLACK, thickness=4)	# Draw bkgd border color before overlaid border color
			cv2.rectangle(img, topLeft, botRight, color=GREEN, thickness=2)
		
		# Convert to RGB for matplotlib
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		axs[idx].imshow(img_rgb)
		axs[idx].axis('off')
		axs[idx].set_title(str_labelsWithProbsFor1Img, fontsize=12, pad=10)

	# Hide unused axes
	for j in range(idx + 1, len(axs)):
		axs[j].axis('off')

	plt.show()



def detectAndRecognize(img_filenames):
	results = []
	for img_filename in img_filenames:
		print(f"\nProcessing file \"{img_filename}\".")

		tic = time.perf_counter()
		result = reader.readtext(img_filename)
		time_elapsed = time.perf_counter() - tic
		print(f"Time to detect and recognize characters (seconds): {time_elapsed:0.3f}")

		results += [result]

		for (bbox, text, prob) in result:
			print(f"Text: \"{text}\",\tProbability={prob:0.2f}")

	visualizeResults_Matplotlib(img_filenames, results)


# Raw strings (r"text-Inside/Raw\String{}[]()") allow usage of single backslashes instead of doubled backslashes
filenames = [r"Datasets\ICDAR2003-sceneDataset\SceneTrialTrain\apanar_06.08.2002\IMG_1247.JPG"]
filenames += [r"Datasets\IIIT5K\test\673_2.png"]
filenames += [r"Datasets\IIIT5K\train\546_4.png"]
print("filenames:",filenames)
detectAndRecognize(filenames)