import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model


# Set page layout to wide
st.set_page_config(layout='wide')

# Set up the sidebar
with st.sidebar:
    st.image('pngwing.png', width=150)
    st.title('LipSync.ai')
    st.info('Trained with the LipNet deep learning model.')

# Main content area
st.title('LipSync')

# Get list of video options
options = os.listdir(os.path.join('.', 'data_female', 's34'))
 
# Select video from dropdown
selected_video = st.selectbox('Choose video', options)

# Create two columns layout
col1, col2 = st.columns(2)

if options:
    # Render the video in the first column
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data_female', 's34', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        
        string_tensor_filepath = tf.constant(selected_video)
        video, annotations = load_data(tf.convert_to_tensor(string_tensor_filepath))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = ''.join([bytes.decode(x) for x in num_to_char(annotations.numpy()).numpy()])
        st.text(converted_prediction)