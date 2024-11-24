import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
import json

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def create_folders():
  folders = ['papers', 'reviews', 'summaries', 'data', 'new_review']
  for folder in folders:
      if not os.path.exists(folder):
          os.makedirs(folder)
          print(f"{folder} 폴더가 생성되었습니다.")

def connect_to_db():
  return mysql.connector.connect(
      host="localhost",
      user="root",
      password=os.getenv('MYSQL_PASSWORD'),
      database="openreview"
  )

def truncate_text(text, max_length=8000):
  if text and len(text) > max_length:
      return text[:max_length] + "..."
  return text

def save_to_file(folder, filename, content):
  filepath = os.path.join(folder, f"{filename}.txt")
  with open(filepath, 'w', encoding='utf-8') as f:
      f.write(content)
  print(f"파일이 저장되었습니다: {filepath}")

def summarize_paper(paper_text):
  truncated_text = truncate_text(paper_text)
  
  llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
  
  prompt = f"""
  Summarize the following academic paper in English. Focus on:
  1. Main contributions and key ideas
  2. Methodology
  3. Main results
  
  Keep the summary under 200 words.
  
  Paper: {truncated_text}
  """
  
  try:
      response = llm.invoke(prompt)
      return response.content
  except Exception as e:
      print(f"요약 중 에러 발생: {e}")
      return "Error summarizing paper"

def get_reviews_data(db):
   try:
       cursor = db.cursor(dictionary=True)
       cursor.execute("""
           SELECT f.id as forum_id, 
                  f.text as paper_text,
                  r.confidence, 
                  r.confidence_reasoning,
                  r.strength_and_weaknesses,
                  r.summary_of_the_paper
           FROM forum f
           JOIN review r ON f.id = r.forum
           LIMIT 100
       """)
       result = cursor.fetchall()
       cursor.close()
       
       summarized_results = []
       for i, row in enumerate(result):
           print(f"\n요약 중... 논문 {i+1}/100")
           if row['paper_text']:
               # 벡터 DB용 데이터는 요약본 사용
               truncated_text = truncate_text(row['paper_text'])
               summary = summarize_paper(truncated_text)
               row['paper_summary'] = summary
               summarized_results.append(row)
               print(f"요약 완료: {summary[:100]}...")
       
       return summarized_results
   except mysql.connector.Error as err:
       print(f"[Mysql] Error: {err}")

def create_vector_db(reviews_data):
  documents = []
  metadatas = []
  
  for review in reviews_data:
      if review.get('paper_summary'):
          documents.append(review['paper_summary'])
          metadatas.append({
              'forum_id': review['forum_id'],
              'confidence': review['confidence'],
              'confidence_reasoning': review['confidence_reasoning'],
              'paper_text': review['paper_text'],  # 전체 텍스트도 메타데이터에 포함
              'strength_and_weaknesses': review['strength_and_weaknesses'],
              'summary_of_the_paper': review['summary_of_the_paper']
          })

  print(f"\n임베딩할 문서 수: {len(documents)}")
  
  embeddings = OpenAIEmbeddings()
  vector_db = Chroma.from_texts(
      documents,
      embeddings,
      metadatas=metadatas,
      persist_directory="./chroma_db"
  )
  
  return vector_db

def save_similar_papers(similar_docs):
   # 유사한 논문들의 정보 저장
   for i, doc in enumerate(similar_docs):
       # 논문 전체 텍스트 저장
       save_to_file('papers', f"similar_paper_{i+1}", doc.metadata['paper_text'])
       
       # 리뷰 정보 저장
       review_content = {
           'confidence': doc.metadata['confidence'],
           'confidence_reasoning': doc.metadata['confidence_reasoning'],
           'strength_and_weaknesses': doc.metadata['strength_and_weaknesses'],
           'summary_of_the_paper': doc.metadata['summary_of_the_paper']
       }
       save_to_file('reviews', f"similar_review_{i+1}", 
                   json.dumps(review_content, indent=2))
       
       # 요약본 저장
       summary = doc.page_content
       save_to_file('summaries', f"similar_summary_{i+1}", summary)

def generate_review(vector_db, new_paper_text):
  print("\n새 논문 요약 중...")
  new_paper_summary = summarize_paper(new_paper_text)
  print(f"새 논문 요약본: {new_paper_summary[:100]}...")

  retriever = vector_db.as_retriever(
      search_type="similarity",
      search_kwargs={"k": 2}
  )

  llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever,
      return_source_documents=True
  )

  query = f"""
  Given this new paper summary: {new_paper_summary}
  
  Based on the 2 most similar papers and their reviews, provide:
  1. A confidence score (1-5)
  2. Detailed reasoning for the confidence score
  
  Format your response as:
  Confidence Score: [score]
  Reasoning: [your detailed explanation]
  """

  result = qa_chain({"query": query})
  
  # 유사한 논문들 정보 저장
  save_similar_papers(result['source_documents'])
  
  # 새 논문의 리뷰 저장
  save_to_file('new_review', 'generated_review', result['result'])
  
  return result

def main():
  db = None
  try:
      create_folders()
      
      print("데이터베이스 연결 중...")
      db = connect_to_db()
      
      print("\n리뷰 데이터 가져오고 논문 요약 중...")
      reviews_data = get_reviews_data(db)
      
      print("\nVector DB 생성 중...")
      vector_db = create_vector_db(reviews_data)
      
      # data 폴더에서 새 논문 읽기
      with open('data/extracted_14284_Beyond_Random_Masking_Wh_20241125_051050.txt', 'r', encoding='utf-8') as f:
          new_paper = f.read()
      
      print("\n리뷰 생성 중...")
      review_result = generate_review(vector_db, new_paper)
      
      print("\n생성된 리뷰:")
      print(review_result['result'])
      
  except Exception as e:
      print(f"에러 발생: {e}")
  finally:
      if db is not None:
          db.close()
          print("\n데이터베이스 연결 종료")

if __name__ == "__main__":
  main()