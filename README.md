**📊 Pneumonia Detection from Chest X-Ray Images using CNN**

This repository contains a deep learning-based image classification model built with TensorFlow and Keras to classify chest X-ray images into three categories: Normal, Bacterial Pneumonia, and Viral Pneumonia. The project demonstrates the application of AI in healthcare diagnostics with a focus on accessibility and real-time inference via Gradio.

**🚀 Project Overview**

Developed and trained a CNN model to detect pneumonia from chest X-ray images.

Preprocessed and augmented medical image data for robust model generalization.

Evaluated performance using validation accuracy and test accuracy metrics.

Deployed the model using Gradio for interactive, real-time image prediction.

**🧐 Motivation**

Manual interpretation of chest X-rays can be slow, error-prone, and requires expert knowledge. This project aims to assist clinicians by providing a fast, accurate, and accessible AI-based diagnostic tool.

**📁 Dataset Details**

Source: Stored on Google Drive (linked in Colab)

Total Images: ~5,306 (Train: 3737, Validation: 935, Test: 634)

**Classes:**

0 = Normal

1 = Bacterial Pneumonia

2 = Viral Pneumonia


**🔧 Technologies Used**

| Category        | Tools/Libraries       |
| --------------- | --------------------- |
| Programming     | Python, Google Colab  |
| Deep Learning   | TensorFlow, Keras     |
| Data Processing | Pandas, NumPy, OpenCV |
| Visualization   | Matplotlib, Seaborn   |
| Deployment      | Gradio                |


**📊 Model Architecture**

Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])


**📊 Training Progress**
Trained for 20 epochs on augmented image data using ImageDataGenerator. The model showed progressive improvement:

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Val Loss |
| ----- | -------------- | ------------------- | ---------- | -------- |
| 1     | 45.8%          | 48.3%               | 1.5891     | 1.0639   |
| 10    | 67.0%          | 73.4%               | 0.7618     | 0.6478   |
| 20    | **72.8%**      | **74.5%**           | 0.6594     | 0.6440   |


**🔬 Test Results**
Test Accuracy: 82.18%

Test Loss: 0.5505

Final model evaluated on 634 test images from 3 categories. Demonstrates strong generalization and predictive power.


**💡 Prediction Demo (Gradio Interface)**
After saving the model (chest_xray_model.h5), a Gradio interface is created:

def predict_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_labels = ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia']
    return class_labels[np.argmax(prediction)]

Run via:

g.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Textbox(label="Prediction")
)


**🔄 Future Work**

1. 🧠 Integrate Explainable AI (XAI) Tools
Why: Trust and transparency are critical in healthcare. Doctors need to understand which parts of the X-ray influenced the model’s decision.

What: Use Grad-CAM to generate heatmaps that show the areas most relevant to each prediction.

How: Overlay these heatmaps on the original X-ray using libraries like tf-keras-vis or OpenCV, and display them within the interface.

2. 🌐 Real-Time Web Deployment
Why: A browser-accessible tool enables clinicians, researchers, and students to use the model without any installation.

What: Convert the model into a web application using Streamlit or Flask.

How: Host the app on platforms like Heroku, Render, or AWS EC2 with user-friendly upload and output display features.

3. 🫁 Expand to Other Thoracic Diseases with Similar Radiological Features
Why: Diseases like COVID-19, tuberculosis, and lung cancer share overlapping features with pneumonia, and misclassification can be risky.

What: Evolve the model into a multi-label classifier that includes these additional conditions.

How: Use datasets from NIH, RSNA, CheXpert, and COVID-Net, and apply transfer learning for model expansion.

4. ⚙️ Automated Data Ingestion Pipeline
Why: For clinical use, X-ray scans should be automatically processed as they are generated or uploaded.

What: Build a pipeline that detects new images in cloud storage or hospital PACS and auto-triggers predictions.

How: Use AWS Lambda triggers, secure APIs, or DICOM-based integration systems to automate end-to-end flow.

5. 🔐 Ensure Data Privacy and Security
Why: Medical data handling must comply with legal and ethical standards like HIPAA and GDPR.

What: Implement encryption, secure authentication, and anonymization practices in data handling and deployment.

How: Use HTTPS APIs, secure token-based access, and avoid storing identifiable metadata in logs or databases.
