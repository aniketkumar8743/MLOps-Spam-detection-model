import pandas as pd
import os
import logging
import string
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# ensure the logs direc 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Pre-processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'Pre-processing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
  """
  tranforms the input text by converting it to lowercase, tokenizing stopwords
  """
  ps = PorterStemmer()
  text = text.lower() # convert into lower case
  text = nltk.word_tokenize(text) # tokenize into words
  text = [word for word in text if word.isalnum()] # remove special word
  text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation] # remove stopwords and punctuation
  text = [ps.stem(word) for word in text] # set steamming
  return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
  try:
    logger.debug('Starting precossing for Dataframe')
    # Encoder the target
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])
    logger.debug('Target Columns encoded')
    
    # Remove duplicated rows
    df = df.drop_duplicates(keep='first')
    logger.debug('Duplicated removed')
    
    # Apply text transformation 
    df.loc[:,text_column] = df[text_column].apply(transform_text)
    logger.debug('text Column Transformed')
    return df
  except KeyError as e:
    logger.error('Column not found: %s', e)
    raise
  except Exception as e:
    logger.error('Error during text normalization: %s', e)
    raise

def main(text_column='text', target_column='target'):
  """
  Main Function to load raw data, preproceesing it , and save the processed data.
  """
  try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.debug('data load')
    
    train_processed_data = preprocess_df(train_data,text_column, target_column)
    test_processed_data = preprocess_df(test_data,text_column, target_column)
    
    # store the data inside data/processed
    data_path = os.path.join("./data","interim")
    os.makedirs(data_path,exist_ok=True)
    
    train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)
    
    logger.debug('Processed data saved to %s',data_path)
  except FileNotFoundError as e:
    logger.error('file not found: %s', e)
  except pd.errors.EmptyDataError as e:
    logger.error('No data: %s', e)
  except Exception as e:
    logger.error('Failed to complete the data transformation process: %s',e)
    print(f'Error: {e}')
    
if __name__ =="__main__":
  main()
    
  
                
  
  
