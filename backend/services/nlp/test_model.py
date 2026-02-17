from sentence_transformers import SentenceTransformer

print("Loading model... (first time may take 1-2 minutes)")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

sentence = "The clouds shone softly in the evening sky."

embedding = model.encode(sentence)

print("Sentence:", sentence)
print("Embedding vector length:", len(embedding))
print("First 5 values:", embedding[:5])
