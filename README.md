1. sources:
Germany: https://adilbari.wordpress.com/wp-content/uploads/2015/07/md-guide-to-driving-in-germany.pdf
https://routetogermany.com/drivingingermany/road-signs#google_vignette --> not free content or redistributable
https://www.gettingaroundgermany.info/zeichen2.shtml --> free data

UK: https://www.gov.uk/browse/driving/highway-code-road-safety


Prompt: i have a pdf containing traffic rules. This pdf has both images and descriptions of traffic rules. I would like to have a llm application which should be trained on this pdf. When asked questions related to traffic rules it should display both the image and description. How to have a training and inference setup

Required:
- Download and install ollama `curl -fsSL https://ollama.com/install.sh | sh`

How to use:
1. Install the requirements - pip install -r requirements.txt
2. Data - "Drivers-Handbook.pdf"
3. Run the jupyter notebook - "working_example.ipynb". It uses ollama embeddinng and LLM model, developed using langchain and Chromadb vector store


