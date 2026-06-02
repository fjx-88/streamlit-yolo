import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 页面标题
st.title("YOLO 模型在线检测")

# 上传图片
uploaded_file = st.file_uploader("上传图片进行检测", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 打开图片
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="上传的图片", use_column_width=True)

    st.write("正在检测…")

    # 加载 YOLO 模型
    from pathlib import Path

    MODEL_PATH = Path(__file__).parent / "car_4.0.pt"
    model = YOLO(str(MODEL_PATH))

    # 模型预测
    results = model.predict(img_array)

    # 可视化结果
    result_img = results[0].plot()  # numpy array
    st.image(result_img, caption="检测结果", use_column_width=True)
