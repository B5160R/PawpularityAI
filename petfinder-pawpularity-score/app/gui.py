import TestModelsPage
import tkinter as tk
import joblib
import torch
from StartPage import StartPage
from TestModelsPage import TestModelsPage
from ExplorationPage import ExplorationPage

TITLE = "Petfinder Pawpularity Score"
GEOMETRY = "650x750"
FONT = ("Helvetica", 16, "bold")
TITLE_ROW = 0
TITLE_COLUMN = 0
TITLE_COLUMN_SPAN = 6
FRAME_ROW = 3
FRAME_COLUMN = 0

class Application(tk.Tk):

    def __init__(self):
        super().__init__()
        self.setup_gui()
        self.setup_base_frame()
        self.load_models()
    
    def setup_gui(self):
        self.title(TITLE)
        self.geometry(GEOMETRY)
        self.resizable(False, False)
        
    def setup_base_frame(self):
        title_label = tk.Label(
            self,
            text=TITLE,
            font=FONT,
        )
        title_label.grid(row=TITLE_ROW, column=TITLE_COLUMN, columnspan=TITLE_COLUMN_SPAN, padx=5, pady=5)

        self.frames = {}
        for PageClass in (StartPage, TestModelsPage, ExplorationPage):
            frame = PageClass(self)
            self.frames[PageClass] = frame
            frame.grid(row=FRAME_ROW, column=FRAME_COLUMN, sticky="nsew")

        self.show_frame(StartPage)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def load_models(self):
        # Score prediction models
        self.score_regression_model = joblib.load("../score_prediction_models/linear_regression_model/pawpularity_regression_model.pkl")
        self.score_ensemble_model = joblib.load("../score_prediction_models/ensemble_model/ensemble_model.pkl")
        self.score_random_forest_model = joblib.load("../score_prediction_models/random_forrest_regressor_model/random_forest_regressor_model.pkl")
        self.score_stacked_classifier_model = joblib.load("../score_prediction_models/stacked_classifier_model/stacked_classifier_model.pkl")
        self.score_nn_model = torch.load("../score_prediction_models/neural_network_model/neural_network_model.pth")
        
        # Feature finding models
        self.is_occlusion_bayes_model = joblib.load("../feature_finding_models/bayes_model/occlusion_bayes_model.pkl")
        self.is_human_rscv_decision_tree_model = joblib.load("../feature_finding_models/rscv_decision_tree_model/random_search_cv_is_human_decision_tree_model.pkl")
        self.is_human_decision_tree_boost_model = joblib.load("../feature_finding_models/decision_tree_model_boost/is_human_decision_tree_model_boost.pkl")
        
        # CNN models
        self.cat_or_dog_model = torch.load("../image_models/cat_or_dog/cat_or_dog_model.pth")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
