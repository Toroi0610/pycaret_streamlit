FROM python:3.7


WORKDIR /app
ADD logo.png /app
ADD app_classification.py /app
ADD app_regression.py /app
ADD streamlit_ui.py /app
ADD requirements.txt /app
ADD base_setting.csv /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copying all files over
COPY logo.png /app
COPY app_classification.py /app
COPY app_regression.py /app
COPY streamlit_ui.py /app
COPY requirements.txt /app
COPY base_setting.csv /app

# Expose port
ENV PORT 8501
ENV PORT 5000

# cmd to launch app when container is run
CMD streamlit run streamlit_ui.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
