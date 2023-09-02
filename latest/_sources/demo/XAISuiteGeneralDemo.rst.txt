XAISuite General Demo
======================
  
.. container:: cell markdown
   :name: 3798025c

   .. rubric:: XAISuite General Demo
      :name: xaisuite-general-demo

.. container:: cell markdown
   :name: 34ab6d88

   Welcome to the XAISuite General Demo. Here you'll find countless
   examples about how to use XAISuite for easier machine learning model
   training, explanation, and explanation comparison.

   First, start by installing the XAISuite library.

   ``!pip install XAISuite==2.8.3``

   Then, import the library. You may need to restart runtime for this to
   work correctly.

.. container:: cell code
   :name: d895f93f

   .. code:: python

      from xaisuite import*

.. container:: cell markdown
   :name: 852677f9

   Run the following code to see a mini doc of XAISuite classes and
   functions.

.. container:: cell code
   :name: 36183c71

   .. code:: python

      help(DataLoader)
      help(DataProcessor)
      help(ModelTrainer)
      help(InsightGenerator)

.. container:: cell markdown
   :name: a236c1c7

   Let's first start with Data Loading. The
   ``xaisuite.dataHandler.DataLoader``\ class allows loading of data
   from different sources.

.. container:: cell code
   :name: 8348bdc4

   .. code:: python

      DataLoader(make_classification)

.. container:: cell code
   :name: a296246d

   .. code:: python

      DataLoader(load_diabetes, return_X_y = True)

.. container:: cell code
   :name: 424e7d58

   .. code:: python

      import numpy as np
      data = np.zeros(20)
      DataLoader(data)

.. container:: cell code
   :name: 2ec98712

   .. code:: python

      import pandas as pd
      data = pd.DataFrame([[1, 2, 3], [3, 4, 5]])
      DataLoader(data)

.. container:: cell markdown
   :name: e43005e3

   Another option: ``DataLoader('path/to/local/file')``

.. container:: cell markdown
   :name: a7c1c655

   DataLoader also has options to specify the variable names, target
   name, categorical variables, etc. If not specified, these values are
   inferred.

   Visualize variables loaded by ``DataLoader``

.. container:: cell code
   :name: 220a3c2f

   .. code:: python

      load_data = DataLoader(make_classification, n_features = 10)
      load_data.plot()

.. container:: cell markdown
   :name: df9867d9

   XAISuiteGeneralDemoOutput1

.. container:: cell markdown
   :name: 812a5335

   For optimal model training, additional processing must be done to the
   data. This is where ``xaisuite.dataHandler.DataProcessor`` comes into
   play. You can process data with default parameters or pass in your
   own transforms (like from the ``sklearn.preprocessing`` library).

.. container:: cell code
   :name: ce8091ca

   .. code:: python

      load_data = DataLoader(make_classification, n_features = 10)
      DataProcessor(load_data)

.. container:: cell code
   :name: b1d7ee4b

   .. code:: python

      load_data = DataLoader(load_diabetes, return_X_y = True)
      DataProcessor(load_data, test_size = 0.1)

.. container:: cell markdown
   :name: e5eb2063

   You can also use transform components, a short example of which is
   given below with placeholders:

   ``from sklearn.preprocessing import StandardScaler``

   ``load_data = DataLoader(load_diabetes, return_X_y = True)``

   ``DataProcessor(load_data, test_size = 0.1, target_transform = "component: TargetTransform()")``

.. container:: cell markdown
   :name: 60902eca

   To train a model, simply do:

.. container:: cell code
   :name: 018ea3c1

   .. code:: python

      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer("SVR", process_data)

.. container:: cell markdown
   :name: afcb63c5

   You can also pass in a model directly without using a String
   representation.

.. container:: cell code
   :name: 67136f59

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer(SVR, process_data, epsilon = 0.2)

.. container:: cell markdown
   :name: 27b54716

   For explaining, simply list the desired explanations.

.. container:: cell code
   :name: 14afcbb5

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer(SVR, process_data, explainers = ["lime", "shap"], epsilon = 0.2)

.. container:: cell markdown
   :name: 49c84fb6

   You can pass in arguments to the explainers:

.. container:: cell code
   :name: d9ef2df0

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer(SVR, process_data, explainers = {"lime": {"feature_selection": "none"}, "shap": {}}, epsilon = 0.2)

.. container:: cell markdown
   :name: 83712ab7

   To access the explanations, use the ``getExplanationsFor``,
   ``getAllExplanations``, or ``getSummaryExplanations`` functions. Use
   ``plotExplanations`` for explanation visualization.

.. container:: cell code
   :name: 0ecb9599

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)
      train_model = ModelTrainer(SVR, process_data, explainers = {"lime": {"feature_selection": "none"}, "shap": {}}, epsilon = 0.2)

      explanations = train_model.getExplanationsFor([]) # Gets all explanations. You can also request explanations for a specific instance
      train_model.plotExplanations("lime", 1) #Display the lime explainer results for the 2nd instance returned by getExplanationsFor()

.. container:: cell markdown
   :name: ef5b80c0

   ``Model score is 0.22886080630718109``

   ``Generating explanations.``

   ``0%|          | 0/45 [00:00<?, ?it/s]``

.. container:: cell markdown
   :name: 15fc20ae

   XAISuiteGeneralDemoOutput2

.. container:: cell markdown
   :name: 0e5c7a3d

   Calculate similarity between explainers using the Shreyan Distance

.. container:: cell code
   :name: 0e18456e

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)
      train_model = ModelTrainer(SVR, process_data, explainers = {"lime": {"feature_selection": "none"}, "shap": {}}, epsilon = 0.2)
      explanations = train_model.getExplanationsFor([])

      insights = InsightGenerator(explanations)
      print(insights.calculateExplainerSimilarity("lime", "shap"))

.. container:: cell markdown
   :name: db5fab4e

   ``Model score is 0.14626289816154203``

   ``Generating explanations.``

   ``0%|          | 0/45 [00:00<?, ?it/s]``

   ``0.8081355932203389``

.. container:: cell markdown
   :name: a975e760

   *NOTE*: For examples using tensorflow or pytorch models, check out
   our other demos.

.. container:: cell code
   :name: 7376a584

   .. code:: python
