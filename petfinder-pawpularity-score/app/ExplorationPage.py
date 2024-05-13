import tkinter as tk

class ExplorationPage(tk.Frame):
	def __init__(self, master):
		tk.Frame.__init__(self, master)
		self.nav_frame = tk.Frame(self)
		self.nav_frame.place(anchor="n", relx=0.5, rely=0, width=self.master.winfo_screenwidth())
		self.base_frame = tk.Frame(self)
		self.base_frame.pack(fill="both", expand=True)
		self.setup_navigation_buttons()
		##self.setup_page()

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
  
	def go_to_start_page(self):
		from StartPage import StartPage
		self.master.show_frame(StartPage)
