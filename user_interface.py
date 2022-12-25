import PySimpleGUI as gui
import os.path
import neural
import cv2

x = neural.NeuralPredictor()
temp_path = "./temp/display_img.png"
gui_font = ("Futura", 15)
gui.theme("DarkAmber")

app_file_types = (("PNG files", "*.png"), ("JPG files", "*.jpg"), 
	("JPEG files", "*.jpeg"), ("BMP files", "*.bmp"))

def update_img(img_path, key, mode = "None"):
	if(mode == "None"):
		os.makedirs("./temp")
		display_img = cv2.resize(cv2.imread(img_path), (100,100))
		cv2.imwrite(temp_path, display_img)
		window[key].update(temp_path)
		os.remove(temp_path)
		os.removedirs("./temp")
	else:
		window[key].update(img_path)

layout = [
	[
		gui.FileBrowse("Поиск", font = gui_font, p = (0,15), 
			key="browser", file_types = app_file_types),
		gui.Text("Выберите изображение", font = gui_font, p = (0,15))
	],
	[
		gui.Button("Подтвердить", p = (0,15), font = gui_font, key = "submit"),
		gui.Text("", p = (0,15), font = gui_font, key = "class"),
	],
	[
		gui.Frame("Входные данные",
		[
			[
				gui.Column([[gui.Image(key="input_img")]], 
					p = (0, 15), justification="center", vertical_alignment = "center"),
				
			]
		], size = (175,175), font = gui_font, border_width = 1),
		gui.Frame("Ответ",
		[
			[
				gui.Column([[gui.Image(key="reference_img")]], 
					p = (0, 15), justification="center", vertical_alignment = "center"),
			]
		], size = (175,175), font = gui_font, border_width = 1)
	],
	[
		gui.Button("Выход", p = (0,15), font = gui_font, key = "exit")
	]
]

window = gui.Window("Классификатор дорожных знаков", layout, size = (1000,600), 
	margins = (50,100), element_justification='c', resizable = False)

while True:
	event, values = window.read()
	if event == "exit" or event == gui.WIN_CLOSED:
		break
	if event == "submit":
		try:
			img_path = values["browser"]
			neural_data = neural.get_data(img_path)
			classification = x.get_classification(neural_data)
			file_name = str(x.get_class_id()).replace('[','').replace(']','')
			ref_img_path = f"./ref_imgs/{file_name}.png"
			window["class"].update(classification)
			update_img(img_path, "input_img")
			update_img(ref_img_path, "reference_img", "ref")
		except Exception as e:
			window["class"].update("e")

window.close()