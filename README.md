README FILE

INTRODUCTION:
This project identifies whether a given text is AI-generated or human-generated. The solution uses advanced NLP techniques in combination with machine learning models to classify text with high accuracy. It combines data preprocessing, similarity analysis, and predictive modeling to achieve high accuracy.

PROBLEM STATEMENT:
With the usage of AI tools, distinguishing between an AI-generated text and a human one has become a critical task. It provides a project that builds a detection system with machine learning models trained on a combination of human-generated-Wikipedia-and AI-generated-LLM outputs-text datasets.

OBJECTIVE:
- Extract content from Wikipedia as human-generated text.
- AI content generation using LLM models such as GPT 2.0, Gemini-1.5 Flash and Llama-3.1.
- Cleaning and preprocessing of both datasets for analysis.
- Conduct similarity checks using cosine similarity.
- Analyzing n-gram patterns and POS tagging to identify unique linguistic sequences.

SOLUTION WORKFLOW:
1. Data Preparation
- Data is retrieved from Wikipedia articles by employing the Wikipedia API.
- Generated Content Tools: Google Generative AI, OpenAI ChatGPT
- Text Preprocessing
	a. Removing special characters and extra spaces and stop words
	b. Text normalization (converting into lowercase, stemming/lemmatization)
	c. Text tokenization (by sentence or words)

2. Exploratory Analysis
- Cosine Similarity: TF-IDF vectorization for calculating the similarity between texts
- N-Gram and POS Analysis: To extract patterns and sequences of parts of speech that both documents may share
- Feature Engineering: Generates features like word count, average sentence length, and lexical diversity for machine learning input.

3. Machine Learning Models
We train and evaluate the Logistic Regression,Random Forest Classifier,Support Vector Machine,Gradient Boosting (XGBoost)
Model	

4. Model Pipeline
- Feature Extraction: Generate features from text: TF-IDF vectors, lexical diversity, etc.
- Train-Test Split: Split the dataset into 80% training and 20% testing subsets.
- Model Training: The models will be trained on a combination of text features and engineered features.
- Evaluation Metrics: Accuracy, precision, recall, and F1-score.

5. Results
Feature Importance: The most important feature contributing to the model performance is,
- TF-IDF score of certain n-grams.
- Sequences of POS tags.
- Text length and lexical diversity.


EXAMPLES:

Input:
- Human Text: A Wikipedia article fetched using the keyword "George Washington."
- AI Text: Content generated using prompts like "Explain George Washington in 10,000 words."

Output:
1.Cosine Similarity Score: 0.87
2.Length of Uncleaned Content:
	Human: 120,000 characters
	AI: 150,000 characters
3.Top N-Grams in AI Text:
	('the', 'united', 'states') → 120 occurrences
4.Top POS N-Grams:
	AI: ('NN', 'VBZ', 'DT') → 150 occurrences
	Human: ('NNP', 'IN', 'NN') → 140 occurrences


SETUP INSTRUCTIONS 

Environment Setup

1.Install the required Python libraries:
   !pip install wikipedia google-generativeai langchain_groq nltk sklearn matplotlib numpy scipy

2.Configure API keys:
- For Google Generative AI, set up an API key using genai.configure(api_key="YOUR_API_KEY").
- For Llama API, set the environment variable or pass the key in code.

3.Download NLTK resources:
	import nltk
	nltk.download('punkt')
	nltk.download('stopwords')
	nltk.download('averaged_perceptron_tagger')


Dataset Links
- Human-Generated Content: Fetched using Wikipedia API.
- AI-Generated Content: Generated using Google Generative AI and LangChain Groq.

Additional Notes
- Ensure you have a stable internet connection for API requests.
- Cosine similarity is calculated using the scikit-learn library.
- N-gram and POS analysis may require additional tuning for large datasets.

