from sentence_transformers import SentenceTransformer, util

def calculate_cosine_similarity_scores(prompt_text, img_descriptions):
    # Load SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode prompt and image descriptive content
    prompt_embedding = model.encode(prompt_text, convert_to_tensor=True)
    image_content_embeddings = model.encode(img_descriptions, convert_to_tensor=True)

    # Calculate cosine-similarities using sentence_transformers library
    cosine_scores = util.cos_sim(prompt_embedding.unsqueeze(0), image_content_embeddings)

    # Extract and return only the scores
    scores_list = cosine_scores[0].cpu().numpy().tolist()
    return scores_list

def keep_top_n_values(values, n=3):
    sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
    top_indices = sorted_indices[:n]
    top_values = [values[i] for i in top_indices]
    return top_indices, top_values