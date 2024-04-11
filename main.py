import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    return img_array

def predict_image(img, model):

    img_array = preprocess_image(img)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    

    class_labels = ['Healthy', 'Doubtful', 'Minimal', 'Moderate', 'Severe']
    result = class_labels[predicted_class]

    return result


model_path = "knee_550.12.h5"

model = load_model(model_path)

st.title("Knee Image Classification")


uploaded_file = st.file_uploader("Choose a knee image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Uploaded Image",width=200)

    if st.button("Predict"):

        result = predict_image(uploaded_file, model)
        st.subheader(f"Prediction: {result}")
        if result == 'Severe':
            st.subheader("Causes:")
            st.write("")
            st.write("Age")
            st.write("As people age, the risk of osteoarthritis increases due to wear and tear on the knee joint.")

            st.write("Previous Joint Injuries")
            st.write("Past knee injuries like ligament tears, meniscal tears, or fractures can increase the likelihood of osteoarthritis by altering joint mechanics and accelerating degeneration.")

            st.write("Obesity")
            st.write("Excess body weight increases stress on knee joints, leading to rapid cartilage wear and tear and is a significant risk factor for osteoarthritis.")

            st.write("Genetics")
            st.write("Genetic factors can influence joint structure and function, affecting susceptibility to osteoarthritis.")

            st.write("Joint Misalignment")
            st.write("Abnormal joint alignment disrupts knee joint biomechanics, leading to uneven cartilage wear and contributing to osteoarthritis development.")

            st.write("Joint Overuse")
            st.write("Repetitive stress on the knee joint, especially in occupations or activities involving frequent kneeling, squatting, or heavy lifting, can accelerate joint degeneration.")

            st.write("Muscle Weakness")
            st.write("Weak or imbalanced knee joint muscles affect stability and contribute to joint degeneration.")

            st.write("Inflammatory Conditions")
            st.write("Inflammatory joint diseases like rheumatoid arthritis can accelerate cartilage breakdown and worsen osteoarthritis severity.")

            st.write("Hormonal Factors")
            st.write("Hormonal changes, such as those during menopause, can influence joint health and increase osteoarthritis risk.")

            st.write("Occupational and Recreational Factors")
            st.write("Certain occupations or activities involving repetitive or high-impact knee movements can increase the risk of osteoarthritis.")
            st.write("Other Medical Conditions")
            st.write("Medical conditions like diabetes, metabolic disorders, and hemochromatosis can affect joint health and increase osteoarthritis risk.")
            st.subheader("Treatments")
            st.write("For individuals with severe knee osteoarthritis, surgery may be a consideration when conservative treatments, such as medication, physical therapy, and injections, have not adequately relieved symptoms or improved joint function. Surgical interventions, including partial or total knee replacement surgery, may be recommended to alleviate pain, restore mobility, and improve overall quality of life. However, the decision to undergo surgery should be made in collaboration with a healthcare professional after a thorough evaluation of the individual's condition, treatment goals, and potential risks and benefits of surgery")
        elif result == 'Moderate':
            st.subheader("Causes")
            
            st.write("")
            st.write("Age")
            st.write("As people age, the risk of osteoarthritis increases due to wear and tear on the knee joint.")

            st.write("Obesity")
            st.write("Excess body weight increases stress on knee joints, leading to rapid cartilage wear and tear and is a significant risk factor for osteoarthritis.")

            st.write("Genetics")
            st.write("Genetic factors can influence joint structure and function, affecting susceptibility to osteoarthritis.")

            st.write("Joint Misalignment")
            st.write("Abnormal joint alignment disrupts knee joint biomechanics, leading to uneven cartilage wear and contributing to osteoarthritis development.")

            st.write("Joint Overuse")
            st.write("Repetitive stress on the knee joint, especially in occupations or activities involving frequent kneeling, squatting, or heavy lifting, can accelerate joint degeneration.")

            st.write("Muscle Weakness")
            st.write("Weak or imbalanced knee joint muscles affect stability and contribute to joint degeneration.")

            st.write("Inflammatory Conditions")
            st.write("Inflammatory joint diseases like rheumatoid arthritis can accelerate cartilage breakdown and worsen osteoarthritis severity.")

            st.write("Hormonal Factors")
            st.write("Hormonal changes, such as those during menopause, can influence joint health and increase osteoarthritis risk.")

            st.write("Occupational and Recreational Factors")
            st.write("Certain occupations or activities involving repetitive or high-impact knee movements can increase the risk of osteoarthritis.")

            st.write("Other Medical Conditions")
            st.write("Medical conditions like diabetes, metabolic disorders, and hemochromatosis can affect joint health and increase osteoarthritis risk.")
            
            st.subheader("Treatments")
            st.write("Medications:")
            st.write("Pain relievers: Over-the-counter acetaminophen or NSAIDs can reduce pain and inflammation.")
            st.write("Topical treatments: Capsaicin or NSAID creams can provide localized pain relief.")
            st.write("Corticosteroid injections: These provide temporary relief from pain and inflammation.")

            st.write("Physical Therapy:")
            st.write("A tailored exercise program designed to strengthen knee muscles, improve flexibility, and enhance joint stability.")
            st.write("Modalities such as ultrasound, electrical stimulation, or hot/cold therapy to reduce pain and inflammation.")

            st.write("Lifestyle Modifications:")
            st.write("Weight management: Losing excess weight reduces load on the knee joints.")
            st.write("Activity modification: Choosing low-impact exercises and avoiding activities that worsen knee pain.")
            st.write("Assistive devices: Using knee braces, orthotic shoe inserts, or walking aids to reduce stress on the knee joint.")

            st.write("Intra-articular Treatments:")
            st.write("Viscosupplementation: Injections of hyaluronic acid to lubricate the joint and reduce pain and stiffness.")
            st.write("Platelet-rich plasma (PRP) therapy: Injections to promote tissue healing and reduce inflammation.")

            st.write("Surgical Interventions:")
            st.write("Arthroscopic debridement: Minimally invasive surgery to remove damaged tissue or smooth irregular joint surfaces.")
            st.write("Osteotomy: Surgical realignment of bones around the knee joint to redistribute weight.")
            st.write("Knee replacement surgery: Total or partial knee replacement surgery in severe cases where conservative treatments have failed.")
        elif result == 'Mild':
            st.subheader("Causes")
            st.write("")
            st.write("Age")
            st.write("As people age, the risk of osteoarthritis increases due to wear and tear on the knee joint.")

            st.write("Previous Joint Injuries")
            st.write("Past knee injuries like ligament tears, meniscal tears, or fractures can increase the likelihood of osteoarthritis by altering joint mechanics and accelerating degeneration.")

            st.write("Obesity")
            st.write("Excess body weight increases stress on knee joints, leading to rapid cartilage wear and tear and is a significant risk factor for osteoarthritis.")

            st.write("Genetics")
            st.write("Genetic factors can influence joint structure and function, affecting susceptibility to osteoarthritis.")

            st.write("Joint Misalignment")
            st.write("Abnormal joint alignment disrupts knee joint biomechanics, leading to uneven cartilage wear and contributing to osteoarthritis development.")

            st.write("Joint Overuse")
            st.write("Repetitive stress on the knee joint, especially in occupations or activities involving frequent kneeling, squatting, or heavy lifting, can accelerate joint degeneration.")

            st.write("Muscle Weakness")
            st.write("Weak or imbalanced knee joint muscles affect stability and contribute to joint degeneration.")

            st.write("Inflammatory Conditions")
            st.write("Inflammatory joint diseases like rheumatoid arthritis can accelerate cartilage breakdown and worsen osteoarthritis severity.")

            st.write("Hormonal Factors")
            st.write("Hormonal changes, such as those during menopause, can influence joint health and increase osteoarthritis risk.")

            st.write("Occupational and Recreational Factors")
            st.write("Certain occupations or activities involving repetitive or high-impact knee movements can increase the risk of osteoarthritis.")

            st.write("Other Medical Conditions")
            st.write("Medical conditions like diabetes, metabolic disorders, and hemochromatosis can affect joint health and increase osteoarthritis risk.")
            
            st.subheader("Treatments")
            st.write("Medications:")
            st.write("Pain relievers: Over-the-counter acetaminophen or NSAIDs can reduce pain and inflammation.")
            st.write("Topical treatments: Capsaicin or NSAID creams can provide localized pain relief.")
            st.write("Corticosteroid injections: These provide temporary relief from pain and inflammation.")

            st.write("Physical Therapy:")
            st.write("A tailored exercise program designed to strengthen knee muscles, improve flexibility, and enhance joint stability.")
            st.write("Modalities such as ultrasound, electrical stimulation, or hot/cold therapy to reduce pain and inflammation.")

            st.write("Lifestyle Modifications:")
            st.write("Weight management: Losing excess weight reduces load on the knee joints.")
            st.write("Activity modification: Choosing low-impact exercises and avoiding activities that worsen knee pain.")
            st.write("Assistive devices: Using knee braces, orthotic shoe inserts, or walking aids to reduce stress on the knee joint.")
        elif result == 'doubtful':
            st.subheader("Causes")
            st.write("")
            st.write("Age")
            st.write("As people age, the risk of osteoarthritis increases due to wear and tear on the knee joint.")

            st.write("Previous Joint Injuries")
            st.write("Past knee injuries like ligament tears, meniscal tears, or fractures can increase the likelihood of osteoarthritis by altering joint mechanics and accelerating degeneration.")

            st.write("Obesity")
            st.write("Excess body weight increases stress on knee joints, leading to rapid cartilage wear and tear and is a significant risk factor for osteoarthritis.")

            st.write("Genetics")
            st.write("Genetic factors can influence joint structure and function, affecting susceptibility to osteoarthritis.")

            st.write("Joint Misalignment")
            st.write("Abnormal joint alignment disrupts knee joint biomechanics, leading to uneven cartilage wear and contributing to osteoarthritis development.")

            st.write("Joint Overuse")
            st.write("Repetitive stress on the knee joint, especially in occupations or activities involving frequent kneeling, squatting, or heavy lifting, can accelerate joint degeneration.")

            st.write("Muscle Weakness")
            st.write("Weak or imbalanced knee joint muscles affect stability and contribute to joint degeneration.")

            st.write("Inflammatory Conditions")
            st.write("Inflammatory joint diseases like rheumatoid arthritis can accelerate cartilage breakdown and worsen osteoarthritis severity.")

            st.write("Hormonal Factors")
            st.write("Hormonal changes, such as those during menopause, can influence joint health and increase osteoarthritis risk.")

            st.write("Occupational and Recreational Factors")
            st.write("Certain occupations or activities involving repetitive or high-impact knee movements can increase the risk of osteoarthritis.")

            st.write("Other Medical Conditions")
            st.write("Medical conditions like diabetes, metabolic disorders, and hemochromatosis can affect joint health and increase osteoarthritis risk.")
            
            st.subheader("Treatments")
            st.write("Medications:")
            st.write("Pain relievers: Over-the-counter acetaminophen or NSAIDs can reduce pain and inflammation.")
            st.write("Topical treatments: Capsaicin or NSAID creams can provide localized pain relief.")
            st.write("Corticosteroid injections: These provide temporary relief from pain and inflammation.")

            st.write("Physical Therapy:")
            st.write("A tailored exercise program designed to strengthen knee muscles, improve flexibility, and enhance joint stability.")
            st.write("Modalities such as ultrasound, electrical stimulation, or hot/cold therapy to reduce pain and inflammation.")

            st.write("Lifestyle Modifications:")
            st.write("Weight management: Losing excess weight reduces load on the knee joints.")
            st.write("Activity modification: Choosing low-impact exercises and avoiding activities that worsen knee pain.")
            st.write("Assistive devices: Using knee braces, orthotic shoe inserts, or walking aids to reduce stress on the knee joint.")
        elif result == 'healthy':
            st.write("your knees are healthy. You don’t have arthritis of the knee.")

def predict_image(img, model):

    img_array = preprocess_image(img)


    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    

    class_labels = ['Healthy', 'Doubtful', 'Minimal', 'Moderate', 'Severe']
    result = class_labels[predicted_class]

    return result

def preprocess_image(img):

    img = image.load_img(img, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    return img_array
