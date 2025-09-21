Flowers classification from kaggle competition

Using a neural network to classify over 50,000 images across 104 flower classes,
with a simple Django web application to predict the class of any input image.

Training was performed on Kaggle Notebooks using a TPU v5e-8.
Dataset provided in .TFREC format by the competition organizers.
TFRecords were parsed and preprocessed (resize + normalize) before being fed to the model. 

Model

Architecture: EfficientNetV2B3 (from Keras library)
Input shape: (224, 224, 3)
Output: GlobalAveragePooling2D + Dropout + Dense layer for classification
Training strategy: Head training followed by fine-tuning

NB: By looking at others project in this kaggle competition is easy to see that the best data setup for this task is to avoid using any
    data augmentation, class weights or label smoothing.I tried all these methods to improve accuracy but achieved almost no improvement. 
    I decided to keep this setup only for didactic purposes.

Log

Applied augmentation to all data:

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),    
        tf.keras.layers.RandomRotation(0.1),         
        tf.keras.layers.RandomZoom(0.1),           
        tf.keras.layers.RandomTranslation(0.1, 0.1),  
        tf.keras.layers.RandomContrast(0.1)         
    ], name="data_augmentation")

    Almost no change (+0.0061 on validation accuracy), likely because the dataset is already large enough.

Added class weights, dataset sample/classes very unbalanced:

    samples_for_class_count= [
                            1088, 104, 80, 84, 2812, 348, 72, 420, 348, 336, 544, 172, 368, 1052,
                            908, 84, 220, 200, 360, 104, 76, 384, 192, 76, 340, 332, 84, 136, 476,
                            436, 420, 96, 92, 80, 72, 144, 228, 104, 76, 292, 256, 384, 252, 440,
                            72, 688, 500, 1044, 1688, 2252, 804, 420, 460, 1840, 148, 232, 356,
                            252, 144, 232, 108, 116, 372, 112, 220, 132, 84, 3128, 1040, 376, 416,
                            548, 668, 1840, 500, 1224, 476, 556, 344, 472, 612, 404, 536, 448,
                            124, 116, 480, 584, 384, 184, 424, 444, 96, 556, 524, 508, 400, 164,
                            136, 96, 124, 100, 1560, 2972
                            ]

    ~3% accuracy drop on validation, but ideally gained accuracy on low samples classes.

Added labels smoothing on top of a different data augmentation for rare classes only:

    Train accuracy:         0.9547
    Validation accuracy :   0.8951

    a little more overfitted, nearly same validation accuracy of the best result but more precision on low samples classes.
    Labels smoothing is useful since differences between some flower classes are very subtle.

#To reproduce training upload kaggle_train.ipynb on a Kaggle notebook.

   Dataset available on this link: https://www.kaggle.com/competitions/flower-classification-with-tpus
   Rename the 4 folder inside flower-classification-with-tpus by removing "tfrecords-jpeg-".
   Only keed the resolution names: "192x192","224x224","331x331","512x512"

   Go to "Datasets" kaggle's page, click on "New Dataset" and upload the dataset.

   Go to "Your Work", click on "Create" and import the kaggle_train.ipynb file.

   In setting enable Internet and set Accelerator to TPU v5e-8 or any available TPU.

   Run all cells in the notebook

#To run django web app on vsc - requires python installed
    
    Reccomended to use a virtual environment. Open a git bash terminal and run:

	python -m venv venv
	source myenv/bin/activate
        
    Install dependencies:

	pip install -r requirements.txt

    Navigate to the app folder:

	cd flower_app/

    Run the Django server:

        py manage.py runserver

    Open your browser and go to:
	
	http://127.0.0.1:8000/

   Enjoy!