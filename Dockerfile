from amr-registry.caas.intel.com/intelaa/python:3.7
COPY . . 
RUN pip install -r req

CMD streamlit run app.py
