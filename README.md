# Age-Detection
The project on Age Detection likely involves using various Python libraries and tools to build a model that predicts the age of individuals from facial images.

1. **Data Acquisition**: I used UTKFace dataset from Kaggle, which contains facial images labeled with age, gender, and ethnicity information. You would download and preprocess this dataset, extracting the facial images and corresponding age labels. Dataset link - https://www.kaggle.com/datasets/jangedoo/utkface-new

2. **Data Preprocessing**: With PIL (Python Imaging Library) and OpenCV (cv), I preprocess the images to standardize them, such as resizing them to a consistent resolution, converting them to grayscale or RGB, and possibly performing data augmentation techniques to increase the diversity of your training dataset.

3. **Model Building**: Using TensorFlow and Keras, I build a deep learning model for age detection. This model might be based on convolutional neural networks (CNNs), which are commonly used for image classification tasks. You could experiment with different architectures, such as VGG, ResNet, or custom architectures tailored to your specific task.

4. **Training**: I split the dataset into training, validation, and possibly test sets. Then, trained the model on the training data.

5. **Deployment**: Using Streamlit, I created a user-friendly interface for your age detection model, allowing users to upload images and receive predictions on the age of individuals in those images. This could involve building a web application or a standalone GUI application, depending on your preferences and requirements.

Overall, my project combines elements of data preprocessing, deep learning model development, training, evaluation, and deployment, leveraging popular Python libraries and tools to create an end-to-end solution for age detection from facial images.


**Two main components for your Age Detection project:**

1. **final.ipynb**: This Jupyter Notebook contains the code for training your age detection model using TensorFlow and Keras. In this notebook, you would preprocess your data, define and compile your deep learning model architecture, train the model on your dataset, and save the trained model weights.

2. **gui.py**: This Python script contains the code for the Streamlit web application that serves as the graphical user interface (GUI) for your age detection model. Users can interact with this GUI to upload images and receive predictions on the age of individuals in those images.

To run the GUI application, you would follow these steps:

1. Open a command prompt or terminal window.
2. Navigate to the directory where the `gui.py` file is located using the `cd` command.
3. Once you're in the correct directory, run the following command:
   ```
   python -m streamlit run gui.py
   ```
   This command tells Python to run the `streamlit` module and execute the `gui.py` script.
4. After running the command, Streamlit will start a local web server and launch the GUI application in your default web browser. You can then use the GUI to upload images and test your age detection model.

Make sure that all necessary dependencies are installed in your Python environment, including Streamlit, TensorFlow, Keras, OpenCV, PIL, and any other libraries you're using in your project.
   ```
   pip install streamlit tensorflow keras opencv-python pillow pandas seaborn matplotlib numpy sklearn
   ```


**Streamlit Output Screen**

**Demo**
https://github.com/DhanyathaShree/Age-Detection/assets/140679630/c528107a-fa27-4b3e-a0ac-79ef01d29f93


![image](https://github.com/DhanyathaShree/Age-Detection/assets/140679630/369623a0-d753-4dc6-b8ed-c18ef64afb4c)

![image](https://github.com/DhanyathaShree/Age-Detection/assets/140679630/f9874d24-2d7b-411c-a916-4c268a90b722)


