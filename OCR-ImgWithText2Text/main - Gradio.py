import easyocr as eo
import cv2
import time

# Look up "EasyOCR language codes" for other language options
reader = eo.Reader(['en'])	# Img contains English ('en') text to be read. This is NOT which language you want the final text result to be. 




def wrapWordsByComma(sentence:str):
	# Word wrap:  Every 4 commas (2 phrases), add a newline to get word wrapping in labels so labels don't clip off the window
	cma_cnt = 0
	for i in range(len(sentence)):
		if sentence[i]==',':
			# Ensure not a numerical comma but rather a syntactical comma
			if (not sentence[i-1].isdigit()) and (not sentence[i+1].isdigit()):
				cma_cnt+=1
				if cma_cnt%4 == 0:
					sentence = sentence[0:i+1]+"\n"+sentence[i+1:]
	return sentence

def createFormattedProbsListOfStr(probs):
	probsTrunc = []
	for prob in probs:
		if prob>0.99:
			# Without the if, cutting off two digits ("1.") yields "00", making user think 0% correct instead of 100% correct
			probsTrunc += ["prob=100%"]
		elif prob<0.1:
			# Turns 0.0999999 into "9%"
			# Truncate all values past 2 decimal pts, then truncate the "0.0" at the start
			probsTrunc += ["prob="+f"{prob:0.2f}"[3:]+"%"]
		else:
			# Truncate all values past 2 decimal pts, then truncate the "0." at the start
			probsTrunc += ["prob="+f"{prob:0.2f}"[2:]+"%"]
	return probsTrunc

def stripSingleQuotesFromProbOnly(s):
	# "('Communication', 'prob=61%'), ('for', 'prob=88%')"  =>  "('Communication', prob=61%), ('for', prob=88%)"
	ret = s[0]
	for i in range(1, len(s)-1):
		if (s[i-1:i+2]==" 'p")  or  (s[i-1:i+2]=="%')"):	continue
		ret += s[i]
	ret += s[-1:]
	return ret

def printToConsole(result, metadata, USE_TOKEN_AND_PROB_OCR):
	print("\n"+metadata)
	if USE_TOKEN_AND_PROB_OCR:
		for (bbox, text, prob) in result:
			print(f"Text: \"{text}\",\tProbability={prob:0.2f}")
	else:
		for (bbox, text) in result:
			print(f"Text: \"{text}\"")


"""
The "multiple images" part of this comment is from an older version that
  supported matplotlib rather than this user-oriented file, but it may be
  useful in the future if parallel processing (CUDA/GPU) is implemented.
Note: EasyOCR and Gradio both support CUDA/multi-processing, so this code
  simply needs to utilize it.

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
# For uploaded img
def getImgWithOverlayAndLabel(img_np, result, USE_TOKEN_AND_PROB_OCR, USE_SMALL_SCREEN):
	if USE_TOKEN_AND_PROB_OCR:
		# print("result:",result)	# [(img1_bbox_list_list, img1_str, img1_prob)]
		# boxes, labels, probs = result
		# ^^^ ValueError: too many values to unpack (expected 3). Causes entire img info to try to be put into boxes,
		#             entire img info to be put into labels, same for probs, and additional imgs are unexpected.
		boxes, labels, probs = zip(*result)

		# Add probabilities to labels
		labelsWithProbs = list(zip(labels, createFormattedProbsListOfStr(probs)))
		str_labelsWithProbs = str(labelsWithProbs)[1:-1]	# Cut off outer "[]"
		str_labelsWithProbs = stripSingleQuotesFromProbOnly(str_labelsWithProbs)
		str_labels = wrapWordsByComma(str_labelsWithProbs)	# Just a name change so that later code works with both tokenized and paragraph-ized code

	else:
		# print("result:",result)	# [(img1_bbox_list_list, img1_str)]
		boxes, str_labels = zip(*result)	# str_labels == paragraph label of every token detected and combined together in the img
	# print("str_labels:",str_labels)
	

	# Sort text from left to right, then top to bottom, which is how English text is read.
	# I used  reader.readtext(img_np, paragraph=True, y_ths = -0.1)  instead of  reader.readtext(img_np),
	#   meaning I don't need to implement this (though it would work perfectly if I implemented it based
	#   on leftmost x coords of boundingbox, then topmost y coords of boundingbox, as opposed to relying
	#   on the OCR reader that doesn't do that by default)
	

	# Draw bounding boxes with a black border
	BLACK, GREEN = (0, 0, 0), (0, 255, 0)
	for (xypair1, xypair2, xypair3, xypair4) in boxes:
		# Low-pixel-thickness borders are nearly invisible on high quality imgs on mobile devices, so scale it up when needed
		thicc_mult = 3 if (len(img_np)>1000 and len(img_np[0])>1000 and USE_SMALL_SCREEN) else 1
		topLeft, botRight = (int(xypair1[0]), int(xypair1[1])),  (int(xypair3[0]), int(xypair3[1])) # Create 2 tuples
		cv2.rectangle(img_np, topLeft, botRight, color=BLACK, thickness=4*thicc_mult)	# Draw bkgd border color before overlaid border color
		cv2.rectangle(img_np, topLeft, botRight, color=GREEN, thickness=2*thicc_mult)

	return (img_np, str(str_labels))


def detectAndRecognize(img_np, USE_TOKEN_AND_PROB_OCR):
	tic = time.perf_counter()
	if USE_TOKEN_AND_PROB_OCR:
		result = reader.readtext(img_np)
	else:
		result = reader.readtext(img_np, paragraph=True, y_ths = -0.1)
	time_elapsed = time.perf_counter() - tic

	metadata = f"Processing uploaded image file of dimensions {len(img_np)}x{len(img_np[0])} and size {img_np.nbytes}."
	metadata += f"\nTime to detect and recognize characters:  {time_elapsed:0.3f} seconds"

	return result, metadata


def gradioHandler(input_img_np, tok_v_para, scrn_size):
	try:
		USE_SMALL_SCREEN       = True  if (len(scrn_size)==1  and scrn_size[0]=="This device has a small screen, so use thicker BBoxes")  else False
		USE_TOKEN_AND_PROB_OCR = True  if (len(tok_v_para)==1 and tok_v_para[0]=="Use token&probability OCR")  else False
		# USE_PARAGRAPH_OCR = not USE_TOKEN_AND_PROB_OCR	# Just for programmer's understanding
		
		result, metadata = detectAndRecognize(input_img_np, USE_TOKEN_AND_PROB_OCR)
		metadata += f"\nUSE_SMALL_SCREEN: {USE_SMALL_SCREEN},\tUSE_TOKEN_AND_PROB_OCR: {USE_TOKEN_AND_PROB_OCR}\n"
		# printToConsole(result, metadata, USE_TOKEN_AND_PROB_OCR)
		img_np, str_labels = getImgWithOverlayAndLabel(input_img_np, result, USE_TOKEN_AND_PROB_OCR, USE_SMALL_SCREEN)
		all_data = metadata + str_labels	# Append imgLabels to metadata
		return img_np, all_data
	except:
		# User deleted the image they previously uploaded onto the site, causing a
		#   change, causing an empty image array to be uploaded as input, causing an exception
		print("LOG:  Empty image array uploaded (likely due to image change/reset).")
		return (None,None)


def visualizeResults_Gradio(USE_INTERNET_ACCESSIBLE_LINK):
	import gradio as gr

	with gr.Blocks() as demo:
		gr.Markdown("# Upload image to view detections")
		with gr.Row():
			input_image = gr.Image(type="numpy", label="Upload Image")
			token_vs_paragraph = gr.CheckboxGroup(["Use token&probability OCR", "Use paragraph OCR"], label="Inference Type (mark exactly 1)")
			scrn_size = gr.CheckboxGroup(["This device has a small screen, so use thicker BBoxes"], label="Bounding Box size")
			output_image = gr.Image(label="Detected & Recognized Image")
		output_label = gr.Textbox(label="Detected & Recognized Words", lines=5, interactive=False)

		input_image.change(
			fn=gradioHandler,
			inputs=[input_image, token_vs_paragraph, scrn_size],
			outputs=[output_image, output_label]
		)

	# share=False == Accessible to your computer only.  share=True == Accessible to world.
	demo.launch(share=USE_INTERNET_ACCESSIBLE_LINK)


USE_INTERNET_ACCESSIBLE_LINK = False	# CHANGE THIS LINE IF YOU WANT
visualizeResults_Gradio(USE_INTERNET_ACCESSIBLE_LINK)





# Raw strings (r"text-Inside/Raw\String{}[]()") allow usage of single backslashes instead of doubled backslashes
# filenames = [r"Datasets\ICDAR2003-sceneDataset\SceneTrialTrain\apanar_06.08.2002\IMG_1247.JPG"]
# filenames += [r"Datasets\IIIT5K\test\673_2.png"]
# filenames = [r"Datasets\IIIT5K\train\546_4.png"]