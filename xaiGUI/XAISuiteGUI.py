#!/usr/bin/env python
# coding: utf-8

#(c) Shreyan Mitra
from tkinter import*
from tkinter import filedialog
from xaisuite import*
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class ABC(Frame):
    def __init__(self,parent=None):
        Frame.__init__(self,parent)
        self.parent = parent
        self.pack()
        self.make_widgets()
        

    def make_widgets(self):
        # don't assume that self.parent is a root window.
        # instead, call `winfo_toplevel to get the root window
        self.winfo_toplevel().title("XAISuite: Training and Explanation Generation Utilities for Machine Learning Models")
        

        # this adds something to the frame, otherwise the default
        # size of the window will be very small
        
        
        self.logo = PhotoImage(file='XAISuiteLogo.png')
        self.values = []
        panel = Label(self, image = self.logo)
        panel.pack(side = "top", fill = "both", expand = "yes")
        welcome = Label(self, text = "Welcome! Please choose one of the following options:", font = 'Helvetica 18 bold')
        welcome.pack(side="top", fill="x")
        button1 = Button(self, text = 'Train and Explain Model', bd = '5',
                          command = self.trainandexplaindialogue)
        button1.pack()
        button2 = Button(self, text = 'Analyze Explanatory Methods', bd = '5',
                          command = self.analyzeexplainersdialogue)
        button2.pack()
        
        
    
    def analyzeexplainersdialogue(self):
        # Toplevel object which will
        # be treated as a new window
        analyzeexplainers = Toplevel()

        # sets the title of the
        # Toplevel widget
        analyzeexplainers.title("New Window")

        # sets the geometry of toplevel
        analyzeexplainers.geometry("200x200")
        
        Label(analyzeexplainers,
              text ="Analyze Explanations", font = 'Helvetica 18 bold').pack()

        
        def openFile():
            self.file_selected = filedialog.askopenfilenames(initialdir=os.getcwd(), title="Select Files for XAISuite", filetypes=[("CSV files", ".csv")])
            filePath.set(self.file_selected)
       
        filePath = StringVar(analyzeexplainers)
        filePath.set("Choose explainer files.")
        Button(analyzeexplainers, text="Choose your data file", command=openFile).pack()
        entry1 = Entry(analyzeexplainers, textvariable=filePath, width=50)
        entry1.pack()
        
            
        
        def core():
            try:
                compare_explanations(list(self.file_selected))
                
            except Exception as e:
                print("Comparison failed: " + str(e))
        
        Button(analyzeexplainers, text="SUBMIT", command= core).pack()
    
    
    def trainandexplaindialogue(self):
        # Toplevel object which will
        # be treated as a new window
        trainandexplain = Toplevel()

        # sets the title of the
        # Toplevel widget
        trainandexplain.title("New Window")

        # sets the geometry of toplevel
        trainandexplain.geometry("200x200")
        
        acceptedModels = {"SVC": "sklearn.svm", "NuSVC": "sklearn.svm", "LinearSVC": "sklearn.svm", "SVR": "sklearn.svm", "NuSVR": "sklearn.svm", "LinearSVR": "sklearn.svm", 
                 "AdaBoostClassifier": "sklearn.ensemble", "AdaBoostRegressor": "sklearn.ensemble", "BaggingClassifier": "sklearn.ensemble", "BaggingRegressor": "sklearn.ensemble",
                 "ExtraTreesClassifier": "sklearn.ensemble", "ExtraTreesRegressor": "sklearn.ensemble", 
                 "GradientBoostingClassifier": "sklearn.ensemble", "GradientBoostingRegressor": "sklearn.ensemble",
                 "RandomForestClassifier": "sklearn.ensemble", "RandomForestRegressor": "sklearn.ensemble",
                 "StackingClassifier": "sklearn.ensemble", "StackingRegressor": "sklearn.ensemble",
                 "VotingClassifier": "sklearn.ensemble", "VotingRegressor": "sklearn.ensemble",
                 "HistGradientBoostingClassifier": "sklearn.ensemble", "HistGradientBoostingRegressor": "sklearn.ensemble",
                 "GaussianProcessClassifier": "sklearn.gaussian_process", "GaussianProcessRegressor": "sklearn.gaussian_process", 
                 "IsotonicRegression": "sklearn.isotonic", "KernelRidge": "sklearn.kernel_ridge", 
                 "LogisticRegression": "sklearn.linear_model", "LogisticRegressionCV": "sklearn.linear_model",
                 "PassiveAgressiveClassifier": "sklearn.linear_model", "Perceptron": "sklearn.linear_model", 
                  "RidgeClassifier": "sklearn.linear_model", "RidgeClassifierCV": "sklearn.linear_model",
                  "SGDClassifier": "sklearn.linear_model", "SGDOneClassSVM": "sklearn.linear_model", 
                  "LinearRegression": "sklearn.linear_model", "Ridge": "sklearn.linear_model", 
                  "RidgeCV": "sklearn.linear_model", "SGDRegressor": "sklearn.linear_model",
                  "ElasticNet": "sklearn.linear_model", "ElasticNetCV": "sklearn.linear_model",
                  "Lars": "sklearn.linear_model", "LarsCV": "sklearn.linear_model", 
                  "Lasso": "sklearn.linear_model", "LassoCV": "sklearn.linear_model",
                  "LassoLars": "sklearn.linear_model", "LassoLarsCV": "sklearn.linear_model",
                  "LassoLarsIC": "sklearn.linear_model", "OrthogonalMatchingPursuit": "sklearn.linear_model",
                  "OrthogonalMatchingPursuitCV": "sklearn.linear_model", "ARDRegression": "sklearn.linear_model",
                  "BayesianRidge": "sklearn.linear_model", "MultiTaskElasticNet": "sklearn.linear_model", 
                  "MultiTaskElasticNetCV": "sklearn.linear_model", "MultiTaskLasso": "sklearn.linear_model",
                  "MultiTaskLassoCV": "sklearn.linear_model", "HuberRegressor": "sklearn.linear_model",
                  "QuantileRegressor": "sklearn.linear_model", "RANSACRegressor": "sklearn.linear_model",
                  "TheilSenRegressor": "sklearn.linear_model", "PoissonRegressor": "sklearn.linear_model",
                  "TweedieRegressor": "sklearn.linear_model", "GammaRegressor": "sklearn.linear_model", 
                  "PassiveAggressiveRegressor": "sklearn.linear_model", "BayesianGaussianMixture": "sklearn.mixture",
                  "GaussianMixture": "sklearn.mixture", 
                  "OneVsOneClassifier": "sklearn.multiclass", "OneVsRestClassifier": "sklearn.multiclass", 
                  "OutputCodeClassifier": "sklearn.multiclass", "ClassifierChain": "sklearn.multioutput", 
                   "RegressorChain": "sklearn.multioutput",  "MultiOutputRegressor": "sklearn.multioutput",
                   "MultiOutputClassifier": "sklearn.multioutput", "BernoulliNB": "sklearn.naive_bayes", 
                  "CategoricalNB": "sklearn.naive_bayes", "ComplementNB": "sklearn.naive_bayes", 
                  "GaussianNB": "sklearn.naive_bayes", "MultinomialNB": "sklearn.naive_bayes", 
                  "KNeighborsClassifier": "sklearn.neighbors", "KNeighborsRegressor": "sklearn.neighbors", 
                  "BernoulliRBM": "sklearn.neural_network", "MLPClassifier": "sklearn.neural_network", "MLPRegressor": "sklearn.neural_network",  
                  "DecisionTreeClassifier": "sklearn.tree", "DecisionTreeRegressor": "sklearn.tree",
                  "ExtraTreeClassifier": "sklearn.tree", "ExtraTreeRegressor": "sklearn.tree"
                 }

        # A Label widget to show in toplevel
        Label(trainandexplain,
              text ="Train and Explain Dialogue", font = 'Helvetica 18 bold').pack()
        
        model = StringVar(trainandexplain)
        model.set("Choose a Model")
        OptionMenu(trainandexplain, model, *acceptedModels.keys()).pack()
        
        def openFile():
            file_selected = filedialog.askopenfilename(title="Select File for XAISuite", filetypes=[("CSV files", ".csv")])
            filePath.set(file_selected)
       
        filePath = StringVar(trainandexplain)
        filePath.set("Choose file above or enter sklearn dataset.")
        target = StringVar(trainandexplain)
        cut = StringVar(trainandexplain)
        index = StringVar(trainandexplain)
        Button(trainandexplain, text="Choose your data file", command=openFile).pack()
        entry1 = Entry(trainandexplain, textvariable=filePath, width=50)
        entry1.pack()
        
       
        target.set("Choose a target variable")
        targetmenu = OptionMenu(trainandexplain, target, ["Placeholder"])
        targetmenu.pack()
        
        cut.set("Enter a cut variable (Optional)")
        cutmenu = OptionMenu(trainandexplain, cut, ["Placeholder"])
        cutmenu.pack()
        
        
        def update_options(*args):
            df = pd.read_csv(filePath.get())
            columns = list(df)
            
            

            options1 = targetmenu['menu']
            options1.delete(0, 'end')
            
            options2 = cutmenu['menu']
            options2.delete(0, 'end')
            

            for variable in columns:
                options1.add_command(label=variable, command=lambda choice=variable: target.set(choice))
                options2.add_command(label=variable, command=lambda choice=variable: cut.set(choice))
        
        filePath.trace('w', update_options)
        
        
        
        
        
        
        Label(trainandexplain,
              text ="Choose Explainers").pack()
        explanations = Listbox(trainandexplain, selectmode = "multiple")
  
        # Widget expands horizontally and
        # vertically by assigning both to 
        # fill option
        explanations.pack()

        # Taking a list 'x' with the items 
        # as explanatory methods
        x = ["shap", "lime", "pdp", "ale"]

        for each_item in range(len(x)):

            explanations.insert(END, x[each_item])

            # coloring alternative lines of listbox
            explanations.itemconfig(each_item,
                     bg = "white")
        
        
        def core():
            if (model.get() == "Choose a Model" or model.get() == "Model is Required"):
                model.set("Model is Required")
                return
            if (len(filePath.get()) == 0 or filePath.get() == "Data file is required." or filePath.get() == "Choose file above or enter sklearn dataset."):
                filePath.set("Data file is required.")
                return
            if (target.get() == "Choose a target variable." or target.get() == "Target Variable Required"):
                target.set("Target Variable Required")
                return
             
            if (cut.get() == "Enter a cut variable."):
                try:
                    train_and_explainModel(model.get(), load_data_CSV(filePath.get(), target.get()), x_ai = self.values) 
                except FileNotFoundError:
                    train_and_explainModel(model.get(), load_data_sklearn(filePath.get(), target.get()), x_ai = self.values) 
                except Exception as e:
                    print("Something went wrong. Check your inputs. " + str(e))
            else:
                try:
                    train_and_explainModel(model.get(), load_data_CSV(filePath.get(), target.get(), cut.get()), x_ai = self.values)
                except FileNotFoundError:
                    train_and_explainModel(model.get(), load_data_sklearn(filePath.get(), target.get()), x_ai = self.values) 
                except Exception as e:
                    print("Something went wrong. Check your inputs. " + str(e))
                    
        
        def callback(event):
            self.values = [event.widget.get(idx) for idx in event.widget.curselection()]
            
        
        Button(trainandexplain, text="SUBMIT", command= core).pack()
            
        
        
        explanations.bind('<<ListboxSelect>>', callback)
        
        
       
        
       
    

        
        

root = Tk()
root.attributes('-fullscreen', True)
abc = ABC(root)
root.mainloop()