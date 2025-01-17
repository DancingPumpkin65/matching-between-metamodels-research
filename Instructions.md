# Instructions  

## Environment Setup (`myenv.zip`)  

To reproduce the results and run the provided codes, you can use the preconfigured environment (`myenv.zip`). This environment includes all the required libraries and dependencies.  

### Steps to Set Up:
1. Download the `myenv.zip` file from [Google Drive](https://drive.google.com/drive/folders/1KeF7nsY-phHfdd_dWbIEIKFhjp1vqVxv?usp=drive_link).  
2. Extract the contents of `myenv.zip`.  
3. Follow the instructions in the extracted folder to activate and use the environment.

### Libraries Used  

The following libraries are used in the codebase and are included in the environment:  

- **Core Libraries:**  
  - `json` (for working with JSON data)  
  - `hashlib` (for hashing utilities)  
  - `pandas` (for data manipulation)  
  - `numpy` (for numerical operations)  

- **Natural Language Processing (NLP):**  
  - `nltk` (for tokenization, stopword removal, and lemmatization)  
    - Modules: `word_tokenize`, `stopwords`, `WordNetLemmatizer`  
  - `sentence-transformers` (for sentence embedding and similarity)  
  - `transformers` (for using BERT tokenizer and model)  
  - `torch` (for PyTorch-based operations)  

- **Similarity Algorithms:**  
  - `simhash` (for Simhash similarity)  

- **Machine Learning:**  
  - `sklearn` (for metrics such as precision, recall, and F-measure)  
  - `scikit-learn` utilities:  
    - `precision_score`  
    - `recall_score`  
    - `f1_score`  
    - `cosine_similarity`  

### Additional Notes:  
- Make sure to have `nltk` data downloaded for tokenization and stopword removal. You can do so by running:  
  ```python  
  import nltk  
  nltk.download('punkt')  
  nltk.download('stopwords')  
  nltk.download('wordnet')  
  ```  
- If any library is missing, you can install it using pip:  
  ```bash  
  pip install pandas sentence-transformers transformers torch simhash scikit-learn  
  ```  
