Tutorials & Example Code
========================

.. container:: cell markdown

   # **XAISuite General Demo**

   .. image:: vertopal_542db4b6b9054ed8a41b9749d9292a67/483c083c2a786f520ac459db067eafc504093a68.png
      :alt: XAISuiteLogo.png

.. container:: cell markdown

   Welcome to the XAISuite General Demo. Here you'll find countless
   examples about how to use XAISuite for easier machine learning model
   training, explanation, and explanation comparison.

   First, start by installing the XAISuite library.

.. container:: cell code

   .. code:: python

      !pip install XAISuite==2.8.3

.. container:: cell markdown

   Then, import the library. You may need to restart runtime for this to
   work correctly.

.. container:: cell code

   .. code:: python

      from xaisuite import*

.. container:: cell markdown

   Run the following code to see a mini doc of XAISuite classes and
   functions.

.. container:: cell code

   .. code:: python

      help(DataLoader)
      help(DataProcessor)
      help(ModelTrainer)
      help(InsightGenerator)

.. container:: cell markdown

   Let's first start with Data Loading. The
   ``xaisuite.dataHandler.DataLoader``\ class allows loading of data
   from different sources.

.. container:: cell code

   .. code:: python

      DataLoader(make_classification)

.. container:: cell code

   .. code:: python

      DataLoader(load_diabetes, return_X_y = True)

.. container:: cell code

   .. code:: python

      import numpy as np
      data = np.zeros(20)
      DataLoader(data)

.. container:: cell code

   .. code:: python

      import pandas as pd
      data = pd.DataFrame([[1, 2, 3], [3, 4, 5]])
      DataLoader(data)

.. container:: cell code

   .. code:: python

      DataLoader('path/to/local/file')

.. container:: cell markdown

   DataLoader also has options to specify the variable names, target
   name, categorical variables, etc. If not specified, these values are
   inferred.

   Visualize variables loaded by ``DataLoader``

.. container:: cell code

   .. code:: python

      load_data = DataLoader(make_classification, n_features = 10)
      load_data.plot()

   .. container:: output display_data

      .. image:: vertopal_542db4b6b9054ed8a41b9749d9292a67/b0152267bd62d16af88d9f9467514291b7042612.png

.. container:: cell markdown

   For optimal model training, additional processing must be done to the
   data. This is where ``xaisuite.dataHandler.DataProcessor`` comes into
   play. You can process data with default parameters or pass in your
   own transforms (like from the ``sklearn.preprocessing`` library).

.. container:: cell code

   .. code:: python

      load_data = DataLoader(make_classification, n_features = 10)
      DataProcessor(load_data)

.. container:: cell code

   .. code:: python

      load_data = DataLoader(load_diabetes, return_X_y = True)
      DataProcessor(load_data, test_size = 0.1)

.. container:: cell code

   .. code:: python

      from sklearn.preprocessing import StandardScaler
      load_data = DataLoader(load_diabetes, return_X_y = True)
      DataProcessor(load_data, test_size = 0.1, target_transform = "component: StandardScaler()")

.. container:: cell markdown

   To train a model, simply do:

.. container:: cell code

   .. code:: python

      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer("SVR", process_data)

.. container:: cell markdown

   You can also pass in a model directly without using a String
   representation.

.. container:: cell code

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer(SVR, process_data, epsilon = 0.2)

.. container:: cell markdown

   For explaining, simply list the desired explanations.

.. container:: cell code

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer(SVR, process_data, explainers = ["lime", "shap"], epsilon = 0.2)

.. container:: cell markdown

   You can pass in arguments to the explainers:

.. container:: cell code

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)

      ModelTrainer(SVR, process_data, explainers = {"lime": {"feature_selection": "none"}, "shap": {}}, epsilon = 0.2)

.. container:: cell markdown

   To access the explanations, use the ``getExplanationsFor``,
   ``getAllExplanations``, or ``getSummaryExplanations`` functions. Use
   ``plotExplanations`` for explanation visualization.

.. container:: cell code

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)
      train_model = ModelTrainer(SVR, process_data, explainers = {"lime": {"feature_selection": "none"}, "shap": {}}, epsilon = 0.2)

      explanations = train_model.getExplanationsFor([]) # Gets all explanations. You can also request explanations for a specific instance
      train_model.plotExplanations("lime", 1) #Display the lime explainer results for the 2nd instance returned by getExplanationsFor()

   .. container:: output stream stdout

      ::

         Model score is 0.22886080630718109
         Generating explanations.

   .. container:: output display_data

      .. code:: json

         {"model_id":"84c355f54f0b4f39a559c591912c0767","version_major":2,"version_minor":0}

   .. container:: output display_data

      .. raw:: html

         <html>
         <head><meta charset="utf-8" /></head>
         <body>
             <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
                 <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>                <div id="96d8ad43-f9b2-4b95-8f4c-3899f393abc1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("96d8ad43-f9b2-4b95-8f4c-3899f393abc1")) {                    Plotly.newPlot(                        "96d8ad43-f9b2-4b95-8f4c-3899f393abc1",                        [{"alignmentgroup":"True","hovertemplate":"Positive=False\u003cbr\u003eImportance scores=%{x}\u003cbr\u003eFeatures=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"False","marker":{"color":"#DC143C","pattern":{"shape":""}},"name":"False","offsetgroup":"False","orientation":"h","showlegend":true,"textposition":"auto","x":[-0.4422319162674623,-0.9095394741836256,-2.2335376637000044],"xaxis":"x","y":["0 = -0.031","1 = 0.051","3 = -0.006"],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"Positive=True\u003cbr\u003eImportance scores=%{x}\u003cbr\u003eFeatures=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"True","marker":{"color":"#008B8B","pattern":{"shape":""}},"name":"True","offsetgroup":"True","orientation":"h","showlegend":true,"textposition":"auto","x":[0.604930019721777,1.4447061707902706,1.759523428903341,1.8156536882616674,4.964694098653471,5.890734263912563,8.071313192297946],"xaxis":"x","y":["9 = 0.003","2 = 0.001","5 = 0.049","4 = 0.064","6 = -0.047","7 = 0.108","8 = 0.084"],"yaxis":"y","type":"bar"}],                        {"barmode":"relative","legend":{"title":{"text":"Positive"},"tracegroupgap":0},"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Instance 1"},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Importance scores"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Features"}}},                        {"responsive": true}                    ).then(function(){
                                     
         var gd = document.getElementById('96d8ad43-f9b2-4b95-8f4c-3899f393abc1');
         var x = new MutationObserver(function (mutations, observer) {{
                 var display = window.getComputedStyle(gd).display;
                 if (!display || display === 'none') {{
                     console.log([gd, 'removed!']);
                     Plotly.purge(gd);
                     observer.disconnect();
                 }}
         }});

         // Listen for the removal of the full notebook cells
         var notebookContainer = gd.closest('#notebook-container');
         if (notebookContainer) {{
             x.observe(notebookContainer, {childList: true});
         }}

         // Listen for the clearing of the current output cell
         var outputEl = gd.closest('.output');
         if (outputEl) {{
             x.observe(outputEl, {childList: true});
         }}

                                 })                };                            </script>        </div>
         </body>
         </html>

.. container:: cell markdown

   Calculate similarity between explainers using the Shreyan Distance

.. container:: cell code

   .. code:: python

      from sklearn.svm import SVR
      load_data = DataLoader(load_diabetes, return_X_y = True)
      process_data = DataProcessor(load_data, test_size = 0.1)
      train_model = ModelTrainer(SVR, process_data, explainers = {"lime": {"feature_selection": "none"}, "shap": {}}, epsilon = 0.2)
      explanations = train_model.getExplanationsFor([])

      insights = InsightGenerator(explanations)
      print(insights.calculateExplainerSimilarity("lime", "shap"))

   .. container:: output stream stdout

      ::

         Model score is 0.14626289816154203
         Generating explanations.

   .. container:: output display_data

      .. code:: json

         {"model_id":"3ebf3bfb23504021a190bf75312d3790","version_major":2,"version_minor":0}

   .. container:: output stream stdout

      ::

         0.8081355932203389

.. container:: cell markdown

   *NOTE*: For examples using tensorflow or pytorch models, check out
   our other demos.

