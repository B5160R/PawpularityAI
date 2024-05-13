import tkinter as tk

class StartPage(tk.Frame):
	def __init__(self, master):
		tk.Frame.__init__(self, master)
		self.nav_frame = tk.Frame(self)
		self.nav_frame.place(anchor="c", relx=5, rely=5)
		self.setup_navigation_buttons()
  
	def setup_navigation_buttons(self):
		tk.Label(self.nav_frame, text="Start").pack(side="left")
		tk.Button(self.nav_frame, text="Test Models", command=self.go_to_test_models_page).pack(side="left")
		tk.Button(self.nav_frame, text="Pattern and Cohesion Exploration", command=self.go_to_exploration_page).pack(side="left")
		self.nav_frame.pack_configure(anchor="center")
	
	def go_to_test_models_page(self):
		from TestModelsPage import TestModelsPage
		self.master.show_frame(TestModelsPage)

	def go_to_exploration_page(self):
		from ExplorationPage import ExplorationPage
		self.master.show_frame(ExplorationPage)
