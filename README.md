# HOW TO USE
## prepare a pdf file 
mkdir -p in && mv *.pdf in/pdf.pdf

## run the script
mkdir -p out out/intermediate_summaries && curl -klf http://localhost:1234/v1/completions | tee /dev/null && python3 summarize-pdf-pipeline.py

# Practically
- aims to summarize a pdf with more than 700 pages
- use a medium to high end GPU
- it took an hour and a half to complete the process

# improvements 
- in and out of file for each response takes 1 min to process. We might be able to make improvements.
