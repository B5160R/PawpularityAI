import tkinter as tk
from PIL import Image, ImageTk

EXPLORATION_DATA_PATHS = {
	"kmeans_clustering_plot": "../data_explorations/unsupervised_pattern_finding/plots/kmeans_clustering.png",
	"pca_visualization_plot": "../data_explorations/unsupervised_pattern_finding/plots/pca_visualization.png",
	"elbow_method_plot": "../data_explorations/unsupervised_pattern_finding/plots/elbow_method.png",
	"db_scan_clustering_plot": "../data_explorations/unsupervised_pattern_finding/plots/dbscan_clustering.png",
	"pawpularity_distribution_plot": "../data_explorations/exploration_outputs/pawpularity_distribution.png",
	"correlation_matrix_plot": "../data_explorations/exploration_outputs/correlation_matrix.png",
	"box_plots": "../data_explorations/exploration_outputs/box_plots.png"
}

class ExplorationPage(tk.Frame):
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
		tk.Button(self.nav_frame, text="Test Models", command=self.go_to_test_models_page).pack(side="left")
		tk.Button(self.nav_frame, text="Pattern and Cohesion Exploration").pack(side="left")
		self.nav_frame.pack_configure(anchor="center")

	def go_to_test_models_page(self):
		from TestModelsPage import TestModelsPage
		self.master.show_frame(TestModelsPage)

	def go_to_start_page(self):
		from StartPage import StartPage
		self.master.show_frame(StartPage)
  
	def setup_page(self):
		button1 = tk.Button(self.base_frame, text="Kmeans Clustering", command=self.show_kmeans_clustering)
		button1.grid(row=0, column=0)
		button2 = tk.Button(self.base_frame, text="PCA Visualization", command=self.show_pca_visualization)
		button2.grid(row=0, column=1)
		button3 = tk.Button(self.base_frame, text="DBSCAN Clustering", command=self.show_db_scan_clustering)
		button3.grid(row=0, column=2)
		button4 = tk.Button(self.base_frame, text="Pawpularity Distribution", command=self.show_pawpularity_distribution)
		button4.grid(row=1, column=0)
		button5 = tk.Button(self.base_frame, text="Correlation Matrix", command=self.show_correlation_matrix)
		button5.grid(row=1, column=1)
		button6 = tk.Button(self.base_frame, text="Elbow Method", command=self.show_elbow_method)
		button6.grid(row=1, column=2)
		button7 = tk.Button(self.base_frame, text="Box Plots", command=self.show_box_plots)
		button7.grid(row=2, column=0)
		result_label = tk.Label(self.base_frame, image=None)
		result_label.grid(row=3, column=0, columnspan=3)
	
	def show_kmeans_clustering(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["kmeans_clustering_plot"])
		image = image.resize((600, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)

	def show_pca_visualization(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["pca_visualization_plot"])
		image = image.resize((600, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)
  
	def show_db_scan_clustering(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["db_scan_clustering_plot"])
		image = image.resize((600, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)
    
	def show_pawpularity_distribution(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["pawpularity_distribution_plot"])
		image = image.resize((800, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)

	def show_correlation_matrix(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["correlation_matrix_plot"])
		image = image.resize((800, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)

	def show_elbow_method(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["elbow_method_plot"])
		image = image.resize((600, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)
  
	def show_box_plots(self):
		self.result_label = tk.Label(self.base_frame, image=None)
		image = Image.open(EXPLORATION_DATA_PATHS["box_plots"])
		image = image.resize((800, 600))
		self.photo = ImageTk.PhotoImage(image)
		self.result_label = tk.Label(self.base_frame, image=self.photo)
		self.result_label.image = self.photo
		self.result_label.grid(row=3, column=0, columnspan=3)