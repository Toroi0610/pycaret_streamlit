FROM python:3.7

WORKDIR /application
ADD /image /application
ADD /app /application
ADD streamlit_ui.py /application
ADD requirements.txt /application

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copying all files over
COPY /image /application
COPY /app /application
COPY streamlit_ui.py /application
COPY requirements.txt /application

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
