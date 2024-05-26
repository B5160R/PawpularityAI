import tkinter as tk
from matplotlib import transforms
import pandas as pd
from PIL import Image, ImageTk
from torchvision import transforms
import torch.nn.functional as functional
import sys
sys.path.append("../image_models/cat_or_dog")
from CatOrDogCNN import CatOrDogCNN

MODEL_PATHS = {
    "Score - Linear Regression": "../score_prediction_models/linear_regression_model/performance_metrics.txt",
    "Score - Ensemble Model": "../score_prediction_models/ensemble_model/performance_metrics.txt",
    "IsHuman - Random Search CV Decision Tree": "../feature_finding_models/decision_tree_model/random_search_cv_performance_metrics.txt",
    "IsOcclusion - Bayes": "../feature_finding_models/bayes_model/performance_metrics.txt",
}

class TestModelsPage(tk.Frame):
	def __init__(self, master):
		tk.Frame.__init__(self, master)
		self.nav_frame = tk.Frame(self)
		self.nav_frame.place(anchor="n", relx=0.5, rely=0, width=self.master.winfo_screenwidth())
		self.base_frame = tk.Frame(self)
		self.base_frame.pack(fill="both", expand=True)
		self.setup_navigation_buttons()
		self.setup_page()

	def setup_navigation_buttons(self):
		tk.Button(self.nav_frame, text="Start", command=self.go_to_start_page).pack(side="left")
		tk.Label(self.nav_frame, text="Test Models").pack(side="left")
		tk.Button(self.nav_frame, text="Pattern and Cohesion Exploration", command=self.go_to_exploration_page).pack(side="left")
		self.nav_frame.pack_configure(anchor="center")
 
	def go_to_start_page(self):
		from StartPage import StartPage
		self.master.show_frame(StartPage)

	def go_to_exploration_page(self):
		from ExplorationPage import ExplorationPage
		self.master.show_frame(ExplorationPage)
  
	def setup_page(self):
		# Checkbox Input Lables and Variables
		checkbox_labels = [
			"Subject Focus",
			"Eyes",
			"Face",
			"Near",
			"Action",
			"Accessory",
			"Group",
			"Collage",
			"Human",
			"Occlusion",
			"Info",
			"Blur",
		]
		self.checkbox_vars = []

		# Create Checkboxes
		for i, label in enumerate(checkbox_labels):
			var = tk.BooleanVar()
			checkbox = tk.Checkbutton(self.base_frame, text=label, variable=var)
			checkbox.grid(row=i // 6 + 1, column=i % 6, padx=5, pady=5)
			self.checkbox_vars.append(var)

		# Create dropdown menu for selecting the model
		models = ["Score - Linear Regression", "Score - Ensemble Model", "IsHuman - RSV Decision Tree", "IsOcclusion - Bayes"]
		self.model_var = tk.StringVar()
		self.model_var.set(models[0])
		model_dropdown = tk.OptionMenu(self.base_frame, self.model_var, *models)
		model_dropdown.grid(row=len(checkbox_labels) // 6 + 2, column=0, padx=5, pady=5, columnspan=2)
	
		# Create Submit Button
		submit_button = tk.Button(self.base_frame, text="Submit", command=self.run_model)
		submit_button.grid(row=len(checkbox_labels) // 6 + 2, column=2, padx=5, pady=5)
	
		# Create Metrics Button
		metrics_button = tk.Button(self.base_frame, text="Show Performance Metrics", command=self.show_metrics)
		metrics_button.grid(row=len(checkbox_labels) // 6 + 2, column=3, padx=5, pady=5)
  
		# Create Result Label
		self.result_label = tk.Label(self.base_frame, text="")
		self.result_label.grid(row=len(checkbox_labels) // 6 + 4, column=0, columnspan=6, padx=5, pady=5)
  
		# Create a Label to Display the IDs of the Pets
		self.pet_id_label = tk.Label(self.base_frame, text="")
		self.pet_id_label.grid(row=len(checkbox_labels) // 6 + 5, column=0, columnspan=6, padx=5, pady=5)

		# Create a Label to Display the Actual Score of the Pet
		self.pet_score_label = tk.Label(self.base_frame, text="")
		self.pet_score_label.grid(row=len(checkbox_labels) // 6 + 6, column=0, columnspan=6, padx=5, pady=5)
  
		# Create Image Label
		self.pet_image_label = tk.Label(self.base_frame)
		self.pet_image_label.grid(row=len(checkbox_labels) // 6 + 7, column=0, columnspan=6, padx=5, pady=5)
	
		# Navigation Functions for results
		self.previous_button = tk.Button(self.base_frame, text="Previous", command=lambda: self.display_result(False))
		self.previous_button.grid(row=len(checkbox_labels) // 6 + 8, column=0, padx=5, pady=5)
		self.previous_button.grid_remove()  # Hide the previous button initially
  
		self.result_number_label = tk.Label(self.base_frame, text="")
		self.result_number_label.grid(row=len(checkbox_labels) // 6 + 8, column=1, padx=5, pady=5)
		
		self.next_button = tk.Button(self.base_frame, text="Next", command=lambda: self.display_result(True))
		self.next_button.grid(row=len(checkbox_labels) // 6 + 8, column=2, padx=5, pady=5)
		self.next_button.grid_remove()  # Hide the next button initially
  
		self.run_cnn_on_image_button = tk.Button(self.base_frame, text="Run CNN on Image", command=self.run_cnn_on_image)
		self.run_cnn_on_image_button.grid(row=len(checkbox_labels) // 6 + 9, column=0, columnspan=6, padx=5, pady=5)
		self.run_cnn_on_image_button.grid_remove()  # Hide the run cnn on image button initially

		self.cnn_result_label = tk.Label(self.base_frame, text="")
		self.cnn_result_label.grid(row=len(checkbox_labels) // 6 + 10, column=0, columnspan=6, padx=5, pady=5)
		self.cnn_performance_label = tk.Label(self.base_frame, text="")
		self.cnn_performance_label.grid(row=len(checkbox_labels) // 6 + 11, column=0, columnspan=6, padx=5, pady=5)

	def run_cnn_on_image(self):
		# Get the pet id
		pet_id = self.pet_ids_and_scores.iloc[self.show_result_index][0]
	
		# Load the image
		image_path = f"../data/pawpularity/train/{pet_id}.jpg"
		image = Image.open(image_path)
	
		# Resize the image
		image = image.resize((64, 64))
	
		# Convert the image to a tensor
		image = transforms.ToTensor()(image)
		image = image.unsqueeze(0)
	
		# Run the image through the CNN
		model = CatOrDogCNN(2) 
		model.load_state_dict(self.master.cat_or_dog_model)
		model.eval()
		output = model(image)
		propabilities = functional.softmax(output, dim=1)
		predicted_class = propabilities.argmax().item()
		certainty_percentage = propabilities[0][predicted_class].item() * 100
  

		# Display the result
		if predicted_class == 0:
			self.cnn_result_label.config(text=f"The model predicts that the image is a cat.\nPrediction Certainty: {certainty_percentage:.2f}%")
		else:
			self.cnn_result_label.config(text=f"The model predicts that the image is a dog.\nPrediction Certainty: {certainty_percentage:.2f}%")

		# Display performance metrics
		metrics = self.read_metrics("../image_models/cat_or_dog/performance_metrics.txt")
		self.cnn_performance_label.config(metrics)
   
	def run_model(self):
		# Get the selected checkbox values
		selected_values = [var.get() for var in self.checkbox_vars]

		# Prepare the input data for the model
		input_data = [int(value) for value in selected_values]
  
		print("--------- Input Data ---------")
		print(input_data)
		print("------------------------------")
  
		# Run the model prediction
		if self.model_var.get() == "Score - Linear Regression":
			prediction = self.master.score_regression_model.predict([input_data])
		elif self.model_var.get() == "Score - Ensemble Model":
			prediction = self.master.score_ensemble_model.predict([input_data])
		elif self.model_var.get() == "IsHuman - Random Search CV Decision Tree":
			prediction = self.master.is_human_rscv_decision_tree_model.predict([input_data])
		elif self.model_var.get() == "IsOcculsion - Bayes":
			prediction = self.master.is_occlusion_bayes_model.predict([input_data])
  
		pet_ids_and_scores = self.get_pet_ids_and_scores(input_data)
		
		if pet_ids_and_scores.empty:
			self.remove_result_labels_and_buttons()
			self.result_label.config(text="No pets match the selected criteria.")
			return
  
		# Add predicted score to pet_indexes
		pet_ids_and_scores["Predicted Score"] = prediction[0]

		self.pet_ids_and_scores = pet_ids_and_scores
  
		# Display results
		self.show_result_index = 0
		self.display_result(False)
  
	def display_result(self, increment):
		if (self.show_result_index >= 0 and increment and self.show_result_index + 2 <= len(self.pet_ids_and_scores)): self.show_result_index += 1
		elif (self.show_result_index > 0 and not increment): self.show_result_index -= 1

		# Clear the CNN result label
		self.cnn_result_label.config(text="")

		i = self.show_result_index
  
		# Display the result number
		self.result_number_label.config(text=f"Showing result: {i+1}/{len(self.pet_ids_and_scores)}")

		# Display the prediction result based on the model
		if self.model_var.get() == "Score - Linear Regression":
			self.result_label.config(text=f"Predicted Pawpularity Score: {self.pet_ids_and_scores.iloc[i][2]} ")
		
		elif self.model_var.get() == "Score - Ensemble Model":
			self.result_label.config(text=f"Predicted Pawpularity Score: {self.pet_ids_and_scores.iloc[i][2]} ")
		
		elif self.model_var.get() == "IsHuman - Decision Tree":
			if self.pet_ids_and_scores.iloc[i][2] == 0:
				self.result_label.config(text="Predicted Is Human: False")
			else:
				self.result_label.config(text="Predicted Is Human: True")

		elif self.model_var.get() == "IsOcclusion - Bayes":
			if self.pet_ids_and_scores.iloc[i][2] == 0:
				self.result_label.config(text="Predicted Is Occluded: False")
			else:
				self.result_label.config(text="Predicted Is Occluded: True")
	
		# Display the pet id
		self.pet_id_label.config(text=f"Pet Id: {self.pet_ids_and_scores.iloc[i][0]}")

		# Display the actual score of the pet
		self.pet_score_label.config(text=f"Actual Pawpularity Score: {self.pet_ids_and_scores.iloc[i][1]}")

		# Display image of the pet with id from the data folder
		pet_image_path = f"../data/pawpularity/train/{self.pet_ids_and_scores.iloc[i][0]}.jpg"
		pet_image = Image.open(pet_image_path)

		# Resize the image
		pet_image = pet_image.resize((300, 300))
		pet_image = ImageTk.PhotoImage(pet_image)

		# Display the image
		self.pet_image_label.config(image=pet_image)
		self.pet_image_label.image = pet_image

		# Display the previous and next buttons
		self.next_button.grid()
		self.previous_button.grid()
  
		# Display the run cnn on image button
		self.run_cnn_on_image_button.grid()
  
	def get_pet_ids_and_scores(self, input_data):
		# Load the train.csv file
		df = pd.read_csv("../data/pawpularity/train.csv")

		# Find the pet indexes that match the input data
		pet_indexes = df[
			(df["Subject Focus"] == input_data[0])
			& (df["Eyes"] == input_data[1])
			& (df["Face"] == input_data[2])
			& (df["Near"] == input_data[3])
			& (df["Action"] == input_data[4])
			& (df["Accessory"] == input_data[5])
			& (df["Group"] == input_data[6])
			& (df["Collage"] == input_data[7])
			& (df["Human"] == input_data[8])
			& (df["Occlusion"] == input_data[9])
			& (df["Info"] == input_data[10])
			& (df["Blur"] == input_data[11])
		].index.tolist()

		return df.loc[pet_indexes, ["Id", "Pawpularity"]]

	def show_metrics(self):
		self.remove_result_labels_and_buttons()
		model_name = self.model_var.get()
		file_path = MODEL_PATHS[model_name]
		metrics = self.read_metrics(file_path)
		self.result_label.config(text=metrics) 

	def read_metrics(self, file_path):
		with open(file_path, "r") as file:
			return file.read()
		
	def remove_result_labels_and_buttons(self):
		self.result_label.config(text="")
		self.pet_id_label.config(text="")
		self.pet_score_label.config(text="")
		self.pet_image_label.config(image="")
		self.result_number_label.config(text="")
		self.next_button.grid_remove()
		self.previous_button.grid_remove()
		self.run_cnn_on_image_button.grid_remove()
		self.cnn_result_label.config(text="")