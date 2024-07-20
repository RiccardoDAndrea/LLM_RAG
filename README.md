# Prime-Chatbot 🤖

Prime-Chatbot is an innovative application that enables users to integrate their own documents as a knowledge source for the Large Language Model (LLM) by using a Retrieval Augmented Generation (RAG) function. The chatbot is based on the powerful GPT-3.5-turbo model and helps to make complex scientific content accessible and understandable.

## Functions

- **Integration of own documents:** Add your documents to provide specific knowledge content for the chatbot.

- **RAG functionality:** Use retrieval augmented generation to get precise and contextualized answers.
- **Complexity reduction:** The chatbot can summarize complicated scientific papers and translate them into simpler language.

## Documentation

### 1. The economic potential of generative AI: The next productivity frontier
Author: McKinsey & Company

This document analyzes the economic potential of generative AI and its transformative impact on various industries. It shows how generative AI can increase productivity, optimize operational processes and create innovative customer experiences. Case studies and examples illustrate the practical applications and benefits of this technology. McKinsey also emphasizes strategic actions that companies should take to realize the full potential of this technology.

### 2. genome network medicine: innovation to overcome huge challenges in cancer therapy
Author: Dimitrios H. Roukos

This paper highlights the importance of genome network medicine in cancer therapy. It shows how next-generation sequencing technologies can be used to identify new biomarkers and therapeutic targets. The paper shows how genome network medicine is helping to overcome the challenges in cancer therapy and advance clinical research. The prime chatbot supports this by simplifying complex terms and providing understandable summaries.

## Usage

### With Docker

If you are familiar with Docker, you can deploy the prime chatbot by running the following command in the terminal or in an IDE:
```
docker pull riccardodandrea/prime-chatbot:tagname
```

### By cloning the repository
Alternatively, you can clone the repository and install the required dependencies:
```
pip install -r requirements.txt
```

Then start the chatbot with:
```
streamlit run Prime-Chatbot.py
```

### Data protection
Confidential PDF documents are neither saved nor cached. However, if there are any concerns, the algorithm is publicly available in the appendix and can be checked.

### Contact
If you have any questions or feedback, please do not hesitate to contact us. We look forward to your feedback and are happy to provide support!


