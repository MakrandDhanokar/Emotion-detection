import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Utils
import pickle
pipe_lr = pickle.load(open('emotion_detection.pkl', 'rb'))


def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]


def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results


emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐",
					   "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Track Utils
# from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

# Main Application

st.title("Emotion Classifier App")



raw_text = st.text_area("Text here")

if st.button('Submit'):

	col1, col2 = st.columns(2)
	prediction = predict_emotions(raw_text)
	probability = get_prediction_proba(raw_text)

	with col1:
		st.success("Original Text")
		st.write(raw_text)
		st.success("Emotion")
		emoji_icon = emotions_emoji_dict[prediction]
		st.write("{}:{}".format(prediction, emoji_icon))
		st.write("Confidence:{}".format(np.max(probability)))

	with col2:
		st.success("Prediction Probability")

		proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
		# st.write(proba_df.T)
		proba_df_clean = proba_df.T.reset_index()
		proba_df_clean.columns = ["emotions", "probability"]

		fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
		st.altair_chart(fig, use_container_width=True)
