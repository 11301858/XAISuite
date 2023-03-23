Want to include a custom model in XAISuite that is not supported? For security issues, we don't just allow any model to be passed to our program. 

Write your model (it must be sklearn compatible) and submit a pull request to add your .py file by pull request to this folder. Your custom model will be reviewed for security issues and XAISuite compatibility. 

If your model is approved, it will be found here and automatically integrated into XAISuite.

By sklearn compatibility, we mean that your model needs to be a sklearn object. If you're using PyTorch or Tensorflow, for example, use wrapper classes. 
